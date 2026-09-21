# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for StagePool.collective_rpc EngineCore control dispatch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture
from vllm.outputs import RequestOutput

from vllm_omni.engine.stage_client import StagePoolLLMClient
from vllm_omni.engine.stage_pool import StagePool, StageUnavailableError
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor

pytestmark = [pytest.mark.core_model]


def _make_pool(*, stage_type: str = "llm", **client_methods: AsyncMock) -> tuple[StagePool, SimpleNamespace]:
    client = SimpleNamespace(stage_type=stage_type, **client_methods)
    if "collective_rpc_async" not in client_methods:
        client.collective_rpc_async = AsyncMock(return_value={"via": "collective"})
    pool = StagePool(0, [client])  # type: ignore[arg-type]
    return pool, client


@pytest.mark.cpu
def test_collective_rpc_normalizes_none_args_on_control_helper():
    async def run() -> None:
        pause = AsyncMock(return_value="paused")
        pool, client = _make_pool(pause_scheduler_async=pause)

        result = await pool.collective_rpc(0, "pause_scheduler", args=None, kwargs={"mode": "abort"})

        assert result == "paused"
        pause.assert_awaited_once_with(mode="abort")
        client.collective_rpc_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_unrelated_async_helper_uses_collective_path():
    async def run() -> None:
        other = AsyncMock(return_value="should-not-run")
        pool, client = _make_pool(reset_prefix_cache_async=other)

        result = await pool.collective_rpc(0, "reset_prefix_cache", timeout=1.5, args=("x",), kwargs={"k": 1})

        assert result == {"via": "collective"}
        other.assert_not_awaited()
        client.collective_rpc_async.assert_awaited_once_with(
            method="reset_prefix_cache",
            timeout=1.5,
            args=("x",),
            kwargs={"k": 1},
        )

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_control_helper_honors_timeout():
    async def run() -> None:
        async def slow_sleep(*_args, **_kwargs):
            await asyncio.sleep(1.0)
            return "slept"

        pool, _client = _make_pool(sleep_async=slow_sleep)
        with pytest.raises(asyncio.TimeoutError):
            await pool.collective_rpc(0, "sleep", timeout=0.01, args=(1,))

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_control_method_reraises_worker_error():
    async def run() -> None:
        pool, _client = _make_pool(wake_up_async=AsyncMock(side_effect=RuntimeError("worker died")))
        with pytest.raises(RuntimeError, match="worker died"):
            await pool.collective_rpc(0, "wake_up")

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_non_control_method_still_returns_error_dict():
    async def run() -> None:
        pool, _client = _make_pool(collective_rpc_async=AsyncMock(side_effect=RuntimeError("probe failed")))
        result = await pool.collective_rpc(0, "reset_prefix_cache")
        assert result["supported"] is False
        assert "probe failed" in result["error"]

    asyncio.run(run())


@pytest.mark.cpu
def test_abort_requests_does_not_commit_op_state_when_engine_abort_fails():
    async def run() -> None:
        class RecordingOutputProcessor:
            def __init__(self) -> None:
                self.collected = False
                self.committed = False

            def abort_requests_collecting_outputs(self, request_ids, *, internal=False, commit_state=True):
                del internal
                self.collected = True
                assert commit_state is False
                return list(request_ids), [SimpleNamespace(request_id=request_ids[0])]

            def commit_aborted_request_state(self, request_ids, *, internal=False):
                del request_ids, internal
                self.committed = True

        abort = AsyncMock(side_effect=RuntimeError("engine abort failed"))
        output_processor = RecordingOutputProcessor()
        client = SimpleNamespace(stage_type="llm", abort_requests_async=abort)
        pool = StagePool(0, [client], output_processor=output_processor)  # type: ignore[arg-type]
        pool._request_bindings["req-1"] = 0

        with pytest.raises(RuntimeError, match="engine abort failed"):
            await pool.abort_requests(["req-1"])
        assert output_processor.collected is True
        assert output_processor.committed is False
        abort.assert_awaited_once()

    asyncio.run(run())


def _make_cleanup_pool(mocker: MockerFixture, *, engine_ids: list[str] | None = None):
    output = RequestOutput(
        request_id="engine-1",
        prompt=None,
        prompt_token_ids=None,
        prompt_logprobs=None,
        outputs=[],
        finished=True,
    )
    processor = mocker.Mock(spec=MultimodalOutputProcessor)
    processor.abort_requests_collecting_outputs.return_value = (
        ["engine-1"] if engine_ids is None else engine_ids,
        [output],
    )
    client = mocker.Mock(spec=StagePoolLLMClient, stage_type="llm")
    client.call_utility_async.return_value = True
    pool = StagePool(0, [client], output_processor=processor)
    pool._request_bindings["req-1"] = 0
    return pool, client, processor, output


@pytest.mark.cpu
@pytest.mark.parametrize("engine_ids", [["engine-1"], []])
def test_prepare_request_cleanup_retains_route_and_noncommitting_snapshot(engine_ids, mocker: MockerFixture):
    pool, client, processor, output = _make_cleanup_pool(mocker, engine_ids=engine_ids)

    plan = pool.prepare_request_cleanup("req-1")

    assert plan.client is client
    assert plan.replica_id == 0
    assert plan.engine_request_ids == (engine_ids or ["req-1"])
    assert plan.engine_request_ids is not engine_ids
    assert plan.abort_outputs == [("req-1", output)]
    assert plan.supported is None
    assert not plan.reclaimed and not plan.released
    processor.abort_requests_collecting_outputs.assert_called_once_with(["req-1"], internal=False, commit_state=False)
    processor.commit_aborted_request_state.assert_not_called()


@pytest.mark.cpu
def test_prepare_request_cleanup_requires_exact_live_binding(mocker: MockerFixture):
    pool, _client, processor, _output = _make_cleanup_pool(mocker)
    with pytest.raises(StageUnavailableError, match="No live cleanup route"):
        pool.prepare_request_cleanup("unknown")
    pool.mark_replica_unavailable(0)
    with pytest.raises(StageUnavailableError, match="No live cleanup route"):
        pool.prepare_request_cleanup("req-1")
    processor.abort_requests_collecting_outputs.assert_not_called()


@pytest.mark.cpu
@pytest.mark.parametrize("phase", ["drain", "reclaim", "release"])
@pytest.mark.parametrize("route_change", ["unavailable", "replaced"])
def test_request_cleanup_never_reselects_original_route(phase, route_change, mocker: MockerFixture):
    async def run() -> None:
        pool, client, _processor, _output = _make_cleanup_pool(mocker)
        plan = pool.prepare_request_cleanup("req-1")
        if phase in {"reclaim", "release"}:
            await pool.drain_request_cleanup(plan, "cancel-1")
        if phase == "release":
            await pool.reclaim_request_cleanup(plan, "cancel-1")
        client.call_utility_async.reset_mock()
        if route_change == "unavailable":
            pool.mark_replica_unavailable(0)
        else:
            pool.clients[0] = mocker.Mock(spec=StagePoolLLMClient, stage_type="llm")
        with pytest.raises(StageUnavailableError, match="Original cleanup route unavailable"):
            await getattr(pool, f"{phase}_request_cleanup")(plan, "cancel-1")
        client.call_utility_async.assert_not_awaited()
        if route_change == "replaced":
            pool.clients[0].call_utility_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize("supported", [False, True])
def test_request_cleanup_reclaim_capability_and_idempotent_phases(supported, mocker: MockerFixture):
    async def run() -> None:
        pool, client, processor, output = _make_cleanup_pool(mocker)
        client.call_utility_async.return_value = supported
        plan = pool.prepare_request_cleanup("req-1")
        await pool.drain_request_cleanup(plan, "cancel-1")
        await pool.drain_request_cleanup(plan, "cancel-1")
        assert plan.supported is supported
        await pool.reclaim_request_cleanup(plan, "cancel-1")
        await pool.reclaim_request_cleanup(plan, "cancel-1")
        assert plan.reclaimed
        await pool.release_request_cleanup(plan, "cancel-1")
        await pool.release_request_cleanup(plan, "cancel-1")
        assert plan.released
        processor.commit_aborted_request_state.assert_not_called()
        pool.commit_request_cleanup(plan)

        expected = [mocker.call("abort_request_and_drain", "req-1", "cancel-1", ["engine-1"])]
        if supported:
            expected.append(mocker.call("reclaim_request_transfer", "req-1", "cancel-1"))
            expected.append(mocker.call("release_request_transfer", "req-1", "cancel-1"))
        assert client.call_utility_async.await_args_list == expected
        assert plan.supported is not None and plan.reclaimed and plan.released
        processor.abort_requests_collecting_outputs.assert_called_once()
        processor.commit_aborted_request_state.assert_called_once_with(["req-1"], internal=False)
        assert plan.abort_outputs == [("req-1", output)]
        assert pool.get_bound_replica_id("req-1") == 0

    asyncio.run(run())


@pytest.mark.cpu
def test_request_cleanup_rejects_unacknowledged_phases_and_invalid_capability(mocker: MockerFixture):
    async def run() -> None:
        pool, client, processor, _output = _make_cleanup_pool(mocker)
        plan = pool.prepare_request_cleanup("req-1")
        with pytest.raises(RuntimeError, match="before drain acknowledgement"):
            await pool.reclaim_request_cleanup(plan, "cancel-1")
        with pytest.raises(RuntimeError, match="before reclamation acknowledgement"):
            await pool.release_request_cleanup(plan, "cancel-1")
        with pytest.raises(RuntimeError, match="before all transfer phases complete"):
            pool.commit_request_cleanup(plan)
        client.call_utility_async.return_value = None
        with pytest.raises(TypeError, match="must return a boolean"):
            await pool.drain_request_cleanup(plan, "cancel-1")
        assert plan.supported is None
        processor.commit_aborted_request_state.assert_not_called()

    asyncio.run(run())
