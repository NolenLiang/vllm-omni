# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Local PR-overlay probe: real Qwen plugin, session runner, fake stage port.

This tests parser/admission composition, not Qwen inference or prompt scheduling.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner import SESSION_ID, RecordingStagePort, _fake_encode_audio
from tests.engine.duplex.test_session_runner_nemotron_voicechat import (
    TIMEOUT_S,
    NemotronHarness,
    error_codes,
    output_item,
)
from tests.engine.duplex.test_tool_call_ledger import FakeTimerScheduler
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.messages import OpenDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.model_channel import _coerce_function_call
from vllm_omni.engine.duplex.session.tool_calls import ToolCallOrigin, ToolCallState
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin
from vllm_omni.protocol.realtime.events import ItemCreated

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@asynccontextmanager
async def qwen_harness():
    plugin = Qwen3OmniDuplexPlugin(_fake_encode_audio)
    plugin.processor = SimpleNamespace()  # No prompt or model execution in this probe.
    port = RecordingStagePort(stage_count=3)
    output = asyncio.Queue()
    results = asyncio.Queue()
    limits = DuplexSessionRuntimeConfig()
    clock = FakeClock(100.0)
    scheduler = FakeTimerScheduler()
    buffer = DuplexOutputBuffer(
        max_bytes=limits.max_pending_output_bytes_per_session, max_events=limits.max_pending_output_events_per_session
    )
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=limits,
        model_config=None,
        clock=clock,
    )
    try:
        config = DuplexSessionConfig(
            model="qwen-integration",
            modalities=["text"],
            extra_body={"auto_response": False, "tool_result_timeout_s": TIMEOUT_S},
        )
        await manager.handle(
            OpenDuplexSessionMessage(
                control_id="open-qwen", session_id=SESSION_ID, session_config=config, output_buffer=buffer
            )
        )
        result = await asyncio.wait_for(results.get(), timeout=2)
        assert result.ok, result
        h = NemotronHarness(
            manager=manager,
            port=port,
            output=output,
            output_buffer=buffer,
            results=results,
            runner=manager.runners[SESSION_ID],
            clock=clock,
            scheduler=scheduler,
        )
        h.runner.tool_calls.set_scheduler(scheduler)
        await h.settle()
        yield h
    finally:
        await manager.shutdown()


async def expose(h, call_id="qwen-call", *, arguments='{"city":"Taipei"}'):
    await h.runner.model._send_one_model_output_event(
        {"qwen_function_call": {"call_id": call_id, "name": "lookup_weather", "arguments": arguments}}
    )
    events = await h.settle()
    assert error_codes(events) == []
    done = next(e for e in events if e.type == "response.function_call_arguments.done")
    call = h.runner.tool_calls.record(call_id)
    assert call is not None and call.state is ToolCallState.PENDING
    assert call.origin.item_id == done.item_id and call.origin.response_id == done.response_id
    assert call.origin.arguments == done.arguments
    assert json.loads(done.arguments) == (json.loads(arguments) if isinstance(arguments, str) else arguments)
    return call


@pytest.mark.asyncio
@pytest.mark.parametrize("forged", [False, True])
async def test_real_qwen_parser_and_prepare_use_session_origin_and_original_ack(forged):
    async with qwen_harness() as h:
        call = await expose(h, arguments={"city": "Taipei"} if forged else '{"city":"Taipei"}')
        hints = {"name": "forged", "arguments": "false"} if forged else {}
        submitted = output_item(call.origin.call_id, **hints)
        events = await h.run(commands.CreateItem(item=submitted, event_id="qwen-result"))
        assert error_codes(events) == [] and call.state is ToolCallState.ACCEPTED
        assert next(e for e in events if isinstance(e, ItemCreated)).item == submitted
        messages = h.session.runtime_config["qwen_tool_messages"]
        assert messages[0]["tool_calls"][0]["function"] == {
            "name": call.origin.name,
            "arguments": call.origin.arguments,
        }
        assert messages[1]["tool_call_id"] == call.origin.call_id
        # Build the real plugin prompt once, without dispatching model work.
        # The acknowledgement's client hints must never become prompt context.
        assert "forged" not in json.dumps(h.session.history)
        captured = []

        def template(prompt_messages, **_kwargs):
            captured.extend(deepcopy(prompt_messages))
            return "prepared-qwen-prompt"

        h.runner.plugin.processor.apply_chat_template = template
        plan = h.runner.plugin.plan_append(
            request_id="probe-followup",
            fence=h.session.fence,
            session_config={"qwen_messages": list(h.session.history)},
            runtime_config=dict(h.session.runtime_config),
            seq=1,
            turn_seq=1,
            payload={"type": "conversation"},
            final=True,
            sampling_params=None,
        )
        assert plan.prompt["prompt"] == "prepared-qwen-prompt"
        captured_call = next(
            tool
            for message in captured
            if message.get("role") == "assistant"
            for tool in message.get("tool_calls", [])
            if tool.get("id") == call.origin.call_id
        )
        assert captured_call["function"] == {
            "name": call.origin.name,
            "arguments": call.origin.arguments,
        }
        captured_result = next(
            message
            for message in captured
            if message.get("role") == "tool" and message.get("tool_call_id") == call.origin.call_id
        )
        assert captured_result["content"] == submitted["output"]
        assert "forged" not in json.dumps(captured)
        before = deepcopy(messages)
        replay = await h.run(commands.CreateItem(item=output_item(call.origin.call_id)))
        assert error_codes(replay) == ["invalid_function_call_output"]
        assert h.session.runtime_config["qwen_tool_messages"] == before


@pytest.mark.asyncio
async def test_real_qwen_preparation_retry_preserves_context_and_deadline(monkeypatch):
    async with qwen_harness() as h:
        call = await expose(h)
        deadline = call.deadline
        real = h.runner.plugin.runtime_config_for_function_output
        seen = []

        def convert(config, current, item):
            seen.append((item["name"], item["arguments"]))
            candidate = real(config, current, item)
            if len(seen) == 1:
                raise DuplexRuntimeConfigError("temporary preparation refusal", code="probe_refused")
            return candidate

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", convert)
        h.clock.advance(1)
        first = await h.run(commands.CreateItem(item=output_item(call.origin.call_id, name="forged-first")))
        assert error_codes(first) == ["probe_refused"] and call.state is ToolCallState.PENDING
        assert call.deadline == deadline and not h.session.runtime_config.get("qwen_tool_messages")
        second = await h.run(commands.CreateItem(item=output_item(call.origin.call_id, name="forged-retry")))
        assert error_codes(second) == [] and call.state is ToolCallState.ACCEPTED
        assert seen == [(call.origin.name, call.origin.arguments)] * 2
        assert len(h.session.runtime_config["qwen_tool_messages"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("timer_first", [False, True])
async def test_real_qwen_expiry_never_prepares_or_installs_late_results(timer_first, monkeypatch):
    async with qwen_harness() as h:
        call = await expose(h)
        real = h.runner.plugin.runtime_config_for_function_output
        prepared = []
        settled = []
        recovered = []
        real_cleanup = h.runner.plugin.on_tool_call_settled
        real_recover = h.runner.plugin.runtime_config_for_tool_call_timeout

        def convert(*args):
            prepared.append(True)
            return real(*args)

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", convert)

        def cleanup(decision):
            settled.append(decision)
            return real_cleanup(decision)

        def recover(config, current, decision):
            recovered.append(decision)
            return real_recover(config, current, decision)

        monkeypatch.setattr(h.runner.plugin, "on_tool_call_settled", cleanup)
        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_tool_call_timeout", recover)
        h.clock.advance(TIMEOUT_S)
        if timer_first:
            h.scheduler.fire_all()
            await h.settle()
        late = await h.run(commands.CreateItem(item=output_item(call.origin.call_id)))
        assert "invalid_function_call_output" in error_codes(late)
        assert call.state is ToolCallState.TIMED_OUT and prepared == []
        assert not h.session.runtime_config.get("qwen_tool_messages")
        timeout_events = [e for e in h.events if e.type == "error" and e.code == "tool_result_timeout"]
        # A timer emits an unsolicited timeout; admission-first replies with a
        # correlated rejection. Both must finalize and notify recovery once.
        assert len(timeout_events) == (1 if timer_first else 0)
        assert len(settled) == len(recovered) == 1
        assert settled[0] == recovered[0] and settled[0].origin == call.origin
        assert settled[0].state is ToolCallState.TIMED_OUT


@pytest.mark.asyncio
async def test_real_qwen_cancel_rejects_old_epoch_result_and_accepts_new_call():
    async with qwen_harness() as h:
        call = await expose(h)
        epoch = h.session.epoch
        await h.run(commands.CancelInput())
        assert h.session.epoch == epoch + 1 and call.state is ToolCallState.CANCELLED
        late = await h.run(commands.CreateItem(item=output_item(call.origin.call_id)))
        assert error_codes(late) == ["invalid_function_call_output"]
        assert not h.session.runtime_config.get("qwen_tool_messages")
        fresh = await expose(h, "qwen-new-call")
        accepted = await h.run(commands.CreateItem(item=output_item(fresh.origin.call_id)))
        assert error_codes(accepted) == [] and fresh.state is ToolCallState.ACCEPTED


@pytest.mark.asyncio
async def test_real_qwen_missing_origin_context_refuses_forged_hints_until_original_expiry():
    async with qwen_harness() as h:
        call = h.runner.tool_calls.register(
            ToolCallOrigin(
                call_id="legacy",
                epoch=h.session.epoch,
                turn_id=h.session.turn_id,
                response_id="legacy-response",
                item_id="legacy-item",
            )
        )
        deadline = call.deadline
        for _ in range(2):
            h.clock.advance(1)
            refused = await h.run(commands.CreateItem(item=output_item("legacy", name="forged", arguments="{}")))
            assert error_codes(refused) == ["invalid_function_call_output"]
            assert call.state is ToolCallState.PENDING and call.deadline == deadline
            assert not h.session.runtime_config.get("qwen_tool_messages")
        h.clock.advance(TIMEOUT_S - 2)
        h.scheduler.fire_all()
        await h.settle()
        assert call.state is ToolCallState.TIMED_OUT


@pytest.mark.parametrize("parser", ["shared", "qwen"])
@pytest.mark.parametrize("arguments", [{"city": "Taipei"}, ["Taipei"], None, 42])
def test_object_call_arguments_remain_json_through_both_recognition_paths(parser, arguments):
    raw = {"call_id": "object-arguments", "name": "lookup", "arguments": arguments}
    if parser == "shared":
        parsed = _coerce_function_call(raw)
    else:
        plugin = Qwen3OmniDuplexPlugin(_fake_encode_audio)
        parsed = plugin.parse_function_call({"qwen_function_call": raw})
    assert parsed is not None and json.loads(parsed["arguments"]) == arguments


@pytest.mark.parametrize("parser", ["shared", "qwen"])
def test_nonserializable_call_arguments_are_not_exposed_as_python_repr(parser):
    raw = {"call_id": "unserializable", "name": "lookup", "arguments": {"opaque"}}
    if parser == "shared":
        parsed = _coerce_function_call(raw)
        assert parsed is None
    else:
        plugin = Qwen3OmniDuplexPlugin(_fake_encode_audio)
        with pytest.raises(DuplexRuntimeConfigError) as error:
            plugin.parse_function_call({"qwen_function_call": raw})
        assert error.value.code == "invalid_function_call"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["legacy", "qwen"])
async def test_nonserializable_recognized_call_is_rejected_before_wire_or_ledger(route):
    async with qwen_harness() as h:
        raw = {"call_id": "invalid-arguments", "name": "lookup", "arguments": {"opaque"}}
        model_result = {"qwen_function_call": raw} if route == "qwen" else {"function_call": True, **raw}
        epoch = h.session.epoch
        await h.runner.model._send_one_model_output_event(model_result)
        events = await h.settle()
        assert error_codes(events) == ["invalid_function_call"]
        assert not any(e.type == "response.function_call_arguments.done" for e in events)
        assert h.runner.tool_calls.record(raw["call_id"]) is None
        assert h.session.epoch == epoch and not h.session.runtime_config.get("qwen_tool_messages")
        assert all("opaque" not in e.message for e in events if e.type == "error")
