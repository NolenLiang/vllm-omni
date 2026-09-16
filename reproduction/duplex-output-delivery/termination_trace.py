# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Temporary [DEBUG-duplex-end] observations for the real websocket probe."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Mapping

_META = (
    "turn_end",
    "end_of_turn",
    "tts_is_last_chunk",
    "duplex_epoch",
    "epoch",
    "duplex_turn_id",
    "turn_id",
)
_MODEL = (
    "data_plane_request_id",
    "model_turn_id",
    "end_of_turn",
    "is_listen",
    "model_listen",
    "is_buffering",
    "prefill_success",
    "stage_role",
    "abort_data_plane_request",
    "requires_stage_handoff",
    "requires_tts_stage",
    "reason",
    "error_code",
)


def _present(value):
    if value is None:
        return False
    numel = getattr(value, "numel", None)
    if callable(numel):
        return numel() > 0
    size = getattr(value, "size", None)
    if isinstance(size, int):
        return size > 0
    try:
        return len(value) > 0
    except TypeError:
        return True


def _scalar(value):
    if isinstance(value, (tuple, list)):
        return _scalar(value[-1]) if value else None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:256]
    # Returned stage metadata may contain small CPU tensors / numpy arrays.
    # Extract one scalar only; never transfer a GPU tensor for instrumentation.
    if getattr(getattr(value, "device", None), "type", "cpu") != "cpu":
        return None
    if hasattr(value, "reshape") and hasattr(value, "item") and _present(value):
        return _scalar(value.reshape(-1)[-1].item())
    return None


def _identity(session):
    return {
        key: getattr(session, key, None)
        for key in (
            "epoch",
            "turn_id",
            "active_request_id",
            "active_response_id",
            "active_response_turn_id",
        )
    }


class TerminationTrace:
    def __init__(self, *, enabled=False, started=None):
        self.enabled = enabled
        self.started = time.monotonic() if started is None else started
        self._rows = deque(maxlen=8192)
        self._counts = Counter()
        self._errors = Counter()
        self._dropped = 0
        self._lock = threading.Lock()
        self._restore = []

    def record(self, phase, session_id=None, **scalars):
        if not self.enabled:
            return
        try:
            row = {
                "at_s": time.monotonic() - self.started,
                "phase": phase,
                "session_id": session_id,
            }
            row.update({key: _scalar(value) for key, value in scalars.items()})
        except Exception:
            self._observation_failed(phase)
            return
        with self._lock:
            self._counts[phase] += 1
            self._dropped += len(self._rows) == self._rows.maxlen
            self._rows.append(row)

    def _observation_failed(self, phase):
        with self._lock:
            if phase not in self._errors and len(self._errors) >= 7:
                phase = "other"
            self._errors[phase] += 1

    def _stage(self, runner, stage_id, output, request_id, context):
        with self._lock:
            self._counts[f"stage_{stage_id}_total"] += 1
        completions = getattr(output, "outputs", None)
        completion = completions[0] if completions else None
        flags = {}
        for source in (
            context.segment_output_metadata,
            getattr(completion, "multimodal_output", None),
            getattr(output, "multimodal_output", None),
        ):
            if not isinstance(source, Mapping):
                continue
            nested = source.get("meta", {})
            for key in _META:
                for mapping, name in (
                    (nested, key),
                    (source, f"meta.{key}"),
                    (source, key),
                ):
                    if isinstance(mapping, Mapping) and name in mapping:
                        value = _scalar(mapping[name])
                        if value is not None:
                            flags[f"metadata_{key}"] = value
        finished = bool(getattr(output, "finished", False))
        if not (
            finished
            or context.segment_finished
            or any(flags.get(f"metadata_{key}") for key in _META[:3])
        ):
            return
        token_ids = getattr(completion, "token_ids", None)
        if token_ids is None:
            token_ids = context.segment_token_ids
        self.record(
            "stage_result",
            context.identity.session_id,
            request_id=request_id,
            stage_id=stage_id,
            final_stage_id=context.final_stage_id,
            output_finished=finished,
            segment_finished=context.segment_finished,
            fence_epoch=context.identity.fence.epoch,
            fence_turn_id=context.identity.fence.turn_id,
            token_count=len(token_ids) if token_ids is not None else 0,
            stop_reason=getattr(completion, "stop_reason", None),
            finish_reason=getattr(completion, "finish_reason", None),
            **flags,
            **_identity(runner.session),
        )

    def install(self):
        if not self.enabled or self._restore:
            return self
        from vllm_omni.engine.duplex.events import ResponseDone
        from vllm_omni.engine.duplex.session.emitter import SessionEmitter
        from vllm_omni.engine.duplex.session.model_channel import ModelChannel
        from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner

        original_stage = DuplexSessionRunner.on_stage_output
        original_model = ModelChannel._send_one_model_output_event
        original_emit = SessionEmitter.emit_events

        def stage(runner, stage_id, output, metrics, *, request_id, context):
            try:
                self._stage(runner, stage_id, output, request_id, context)
            except Exception:
                self._observation_failed("stage_result")
            return original_stage(
                runner,
                stage_id,
                output,
                metrics,
                request_id=request_id,
                context=context,
            )

        async def model(channel, model_result, *, expected_epoch=None):
            fields = {}
            try:
                session = channel._ctx.session
                fields = {key: _scalar(model_result.get(key)) for key in _MODEL}
                fields.update(
                    expected_epoch=expected_epoch,
                    has_text=_present(model_result.get("text")),
                    has_audio=_present(
                        model_result.get("audio_data", model_result.get("audio"))
                    ),
                )
                self.record(
                    "model_result_before",
                    session.session_id,
                    **fields,
                    **_identity(session),
                )
            except Exception:
                self._observation_failed("model_result_before")
            returned = False
            result = None
            try:
                result = await original_model(
                    channel, model_result, expected_epoch=expected_epoch
                )
                returned = True
                return result
            finally:
                try:
                    session = channel._ctx.session
                    self.record(
                        "model_result_after",
                        session.session_id,
                        **fields,
                        **_identity(session),
                        returned=returned,
                        emitted_response=result[1] if returned else None,
                    )
                except Exception:
                    self._observation_failed("model_result_after")

        def emit(emitter, events):
            try:
                session = emitter._ctx.session
                for event in events:
                    if isinstance(event, ResponseDone):
                        self.record(
                            "response_done_projected",
                            session.session_id,
                            epoch=session.epoch,
                            response_id=event.response_id,
                            event_id=event.event_id,
                            status=event.status,
                        )
            except Exception:
                self._observation_failed("response_done_projected")
            return original_emit(emitter, events)

        for cls, name, original, wrapper in (
            (DuplexSessionRunner, "on_stage_output", original_stage, stage),
            (ModelChannel, "_send_one_model_output_event", original_model, model),
            (SessionEmitter, "emit_events", original_emit, emit),
        ):
            self._restore.append((cls, name, original))
            setattr(cls, name, wrapper)
        return self

    def close(self):
        for cls, name, original in reversed(self._restore):
            setattr(cls, name, original)
        self._restore.clear()

    def snapshot(self, roles):
        with self._lock:
            rows, counts, dropped = list(self._rows), dict(self._counts), self._dropped
            errors = dict(self._errors)
        return {
            "tag": "[DEBUG-duplex-end]",
            "enabled": self.enabled,
            "counts": counts,
            "dropped": dropped,
            "instrumentation_errors": sum(errors.values()),
            "instrumentation_errors_by_phase": errors,
            "rows": [{**row, "role": roles.get(row["session_id"])} for row in rows],
        }
