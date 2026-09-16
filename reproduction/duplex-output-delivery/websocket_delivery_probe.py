# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real MiniCPM-o and production duplex handler over loopback WebSocket.

This is not the complete OpenAI app. ``network`` pauses client reads only;
``held`` additionally holds one session's production outbound lock before
sequencing its second audio event. No bytes already sent can be recalled.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import time
from contextlib import AsyncExitStack
from pathlib import Path

from termination_trace import TerminationTrace


async def observe(omni, args):
    import uvicorn
    import websockets
    from fastapi import FastAPI, WebSocket

    from vllm_omni.clients.duplex import SessionConfig
    from vllm_omni.engine.duplex.events import AudioDelta, ResponseDone
    from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler

    started = time.monotonic()
    trace = TerminationTrace(enabled=args.trace_termination, started=started)
    trace.install()
    events, server_events, tasks = [], [], []
    first = {role: asyncio.Event() for role in ("slow", "normal")}
    opened = {role: asyncio.Event() for role in first}
    resume, cancelled, held = asyncio.Event(), asyncio.Event(), asyncio.Event()
    normal_done = asyncio.Event()
    roles, identities, responses, sockets, send_locks = {}, {}, {}, {}, {}
    summary = {
        "status": "running",
        "mode": args.mode,
        "completion_mode": args.completion_mode,
        "cancel_slow": args.cancel_slow,
        "audio_event_type": AudioDelta.wire_type,
        "deploy_config_argument": "pathlib.Path; avoids the upstream 8b028c9 string-path NameError",
        "scope": "real MiniCPM-o; production duplex handler; two loopback WebSockets; not complete OpenAI app",
        "note": "Client pauses before intervention and 2s after it. Omit mode sends no cancellation; it is not cancellation evidence. Observation time is not engine cancel latency. ASGI acceptance is not physical playback.",
        "deliberate_pause_before_cancel_s": args.pause_before_s,
        "deliberate_pause_after_cancel_s": 2,
        "initial_text_repetitions": {
            "slow": args.text_repetitions,
            "normal": args.text_repetitions,
        },
        "errors": [],
        "unexpected_closes": [],
    }
    held_lock = None
    held_event = None
    closing = False

    def now():
        return time.monotonic() - started

    def mark_cleanup():
        nonlocal closing
        closing = True
        summary.setdefault("cleanup_started_at_s", now())

    def compact(payload, role, phase):
        kind = payload.get("type")
        response = payload.get("response", {})
        delta = payload.get("delta", "")
        audio = kind == AudioDelta.wire_type
        audio_data = base64.b64decode(delta, validate=True) if audio and delta else b""
        identity = identities.get(payload.get("event_id"), {})
        return {
            "at_s": now(),
            "role": role,
            "phase": phase,
            "type": kind,
            "event_id": payload.get("event_id"),
            "server_event_seq": payload.get("server_event_seq"),
            "response_id": payload.get("response_id") or response.get("id"),
            "response_status": response.get("status"),
            "epoch": identity.get("epoch"),
            "audio_bytes": len(audio_data),
            "audio_sha256": hashlib.sha256(audio_data).hexdigest()
            if audio_data
            else None,
            "audio_format": payload.get("format") if audio else None,
            "sample_rate_hz": payload.get("sample_rate_hz") if audio else None,
            **({"error": payload.get("error")} if kind == "error" else {}),
        }

    class ObservedHandler(OmniDuplexSessionHandler):
        async def _open(self, websocket, envelope, session_payload, send_json):
            attachment = await super()._open(
                websocket, envelope, session_payload, send_json
            )
            if attachment is not None:
                roles[attachment.handle.session_id] = websocket.query_params["role"]
            return attachment

        async def _send_event(self, session_id, event, **kwargs):
            nonlocal held_lock, held_event
            if isinstance(event, ResponseDone):
                trace.record(
                    "response_done_frontend_consumed",
                    session_id,
                    response_id=event.response_id,
                    event_id=event.event_id,
                    epoch=event.epoch,
                    status=event.status,
                )
            identities[event.event_id] = {
                "epoch": event.epoch,
                "session_id": session_id,
            }
            role = roles.get(session_id)
            if role == "slow" and isinstance(event, AudioDelta):
                summary["slow_server_audio_attempts"] = (
                    summary.get("slow_server_audio_attempts", 0) + 1
                )
                if args.mode == "held" and summary["slow_server_audio_attempts"] == 2:
                    # The actual production send_event now waits on its own lock.
                    # Cancellation can invalidate the already-popped event while
                    # it has neither a sequence number nor a journal entry.
                    held_lock = self._attachment_registry._sessions[
                        session_id
                    ].outbound_lock
                    await held_lock.acquire()
                    held_event = event
                    summary["held_event_id"] = event.event_id
                    summary["held_event_epoch"] = event.epoch
                    summary["held_response_id"] = event.response_id
                    summary["held_at_s"] = now()
                    held.set()
            await super()._send_event(session_id, event, **kwargs)
            if event is held_event:
                state = self._attachment_registry._sessions.get(session_id)
                # Observe immediately: a later snapshot may have expired the
                # entry and cannot establish whether it was ever recorded.
                summary["held_event_in_journal"] = (
                    any(
                        e.payload.get("event_id") == event.event_id
                        for e in state.journal._entries
                    )
                    if state is not None
                    else None
                )
                summary["held_journal_observed_at_s"] = now()

    handler = ObservedHandler(duplex_omni=omni)
    app = FastAPI()

    # FastAPI resolves annotations through globals, not this function's locals.
    async def route(websocket):
        await handler.handle_realtime_session(websocket)

    route.__annotations__["websocket"] = WebSocket
    app.websocket("/v1/realtime")(route)

    async def observed_app(scope, receive, send):
        query = scope.get("query_string", b"").decode()
        role = "slow" if "role=slow" in query else "normal"

        async def observed_send(message):
            await send(message)
            if message["type"] == "websocket.send" and message.get("text"):
                server_events.append(
                    compact(json.loads(message["text"]), role, "asgi_send_accepted")
                )

        await app(scope, receive, observed_send)

    server = uvicorn.Server(
        uvicorn.Config(
            observed_app,
            host="127.0.0.1",
            port=args.port,
            log_level="warning",
            ws="websockets",
            ws_ping_interval=None,
            timeout_graceful_shutdown=5,
        )
    )
    server_task = asyncio.create_task(server.serve())

    async def send(role, payload):
        async with send_locks[role]:
            await sockets[role].send(json.dumps(payload))

    async def feed(role):
        await opened[role].wait()
        for _ in range(600):
            await send(
                role,
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(bytes(6400)).decode(),
                    "format": "pcm16",
                    "sample_rate_hz": 16000,
                    "is_speech": False,
                    "duration_ms": 200,
                },
            )
            await asyncio.sleep(0.2)

    async def receive(role):
        try:
            async for raw in sockets[role]:
                payload = json.loads(raw)
                entry = compact(payload, role, "client_receive")
                events.append(entry)
                if entry["type"] == "session.created":
                    opened[role].set()
                if entry["type"] == "error":
                    summary["errors"].append(entry)
                if (
                    entry["type"] in {"session.closed", "session.expired"}
                    and not closing
                ):
                    summary["unexpected_closes"].append(entry)
                if entry["audio_bytes"] and not first[role].is_set():
                    responses[role] = entry["response_id"]
                    summary[f"{role}_first_audio_at_s"] = now()
                    first[role].set()
                    if role == "slow":
                        await resume.wait()
                if (
                    role == "slow"
                    and entry["response_id"] == responses.get(role)
                    and entry["type"] == "response.done"
                    and entry["response_status"] == "cancelled"
                ):
                    summary["cancelled_terminal_at_s"] = entry["at_s"]
                    cancelled.set()
                if (
                    role == "normal"
                    and entry["response_id"] == responses.get(role)
                    and entry["type"] == "response.done"
                ):
                    normal_done.set()
        except websockets.exceptions.ConnectionClosed as exc:
            if not closing:
                summary["unexpected_closes"].append(
                    {"role": role, "at_s": now(), "message": str(exc)}
                )
        finally:
            if not closing:
                summary["unexpected_closes"].append(
                    {"role": role, "at_s": now(), "message": "receive loop ended"}
                )

    def buffer_snapshot():
        result = {}
        for sid, role in roles.items():
            handle = omni.get_session(sid)
            if handle is not None:
                count = getattr(handle._outbox, "pending_events", None)
                result[role] = {
                    "closed": handle.closed,
                    "pending_events": count
                    if count is not None
                    else handle._outbox.qsize(),
                    "pending_bytes": getattr(handle._outbox, "pending_bytes", None),
                }
        return result

    try:
        async with asyncio.timeout(240), AsyncExitStack() as stack:
            while not server.started:
                if server_task.done():
                    await server_task
                    raise RuntimeError("WebSocket server did not start")
                await asyncio.sleep(0.05)
            story = (
                "Please read this story aloud. A scientist walked through a quiet garden on a clear morning. "
                "She stopped to examine each flower and carefully record its colour, shape, and scent in her notebook. "
                "After an hour she sat beside the pond and began writing a letter to her friend. "
            )
            for role in ("slow", "normal"):
                uri = f"ws://127.0.0.1:{args.port}/v1/realtime?duplex=1&autostart=0&role={role}"
                sockets[role] = await stack.enter_async_context(
                    websockets.connect(
                        uri,
                        max_size=8 * 1024 * 1024,
                        max_queue=1,
                        ping_interval=None,
                    )
                )
                send_locks[role] = asyncio.Lock()
                config = SessionConfig(
                    instructions="Read the supplied text aloud in English, exactly as written.",
                    overlap_policy="listen_only",
                    playback_commit_policy="ack_only",
                    extra_body={
                        "auto_response": True,
                        "duplex_initial_user_text": story
                        * summary["initial_text_repetitions"][role],
                        "force_listen_count": 0,
                    },
                ).to_session_payload(model=args.model)
                ref = args.ref_audio or Path(args.model) / "assets" / "HT_ref_audio.wav"
                config["ref_audio"] = (
                    "data:audio/wav;base64,"
                    + base64.b64encode(ref.read_bytes()).decode()
                )
                await send(role, {"type": "session.update", "session": config})
                tasks.extend(
                    (
                        asyncio.create_task(receive(role)),
                        asyncio.create_task(feed(role)),
                    )
                )
            # Mark failure unwind before AsyncExitStack closes either socket.
            # The original timeout still fails; cleanup is not a spontaneous disconnect.
            stack.callback(mark_cleanup)
            await asyncio.gather(*(event.wait() for event in first.values()))
            if args.mode == "held":
                await held.wait()
            summary["pause_window_started_at_s"] = now()
            await asyncio.sleep(args.pause_before_s)
            if normal_done.is_set():
                raise RuntimeError(
                    "Normal response ended before intervention; this run cannot observe interference."
                )
            summary["before_intervention"] = buffer_snapshot()
            summary["intervention_at_s"] = now()
            if args.cancel_slow == "request":
                summary["cancel_submitted_at_s"] = summary["intervention_at_s"]
                await send(
                    "slow",
                    {"type": "response.cancel", "response_id": responses["slow"]},
                )
            await asyncio.sleep(2)
            summary["after_intervention_before_release"] = buffer_snapshot()
            if held_event is not None:
                handle = omni.get_session(held_event.session_id)
                if handle is not None and hasattr(handle, "output_guard"):
                    with handle.output_guard(held_event) as valid:
                        summary["held_event_valid_before_release"] = valid
            if held_lock is not None:
                held_lock.release()
                held_lock = None
            summary["resume_reads_at_s"] = now()
            resume.set()
            if args.cancel_slow == "request":
                await asyncio.wait_for(cancelled.wait(), timeout=60)
                summary["delivery_window_started_at_s"] = summary[
                    "cancelled_terminal_at_s"
                ]
            else:
                summary["delivery_window_started_at_s"] = summary["resume_reads_at_s"]
            await asyncio.sleep(args.observe_after_s)
            summary["delivery_window_ended_at_s"] = now()
            if args.completion_mode == "full":
                await asyncio.wait_for(normal_done.wait(), timeout=120)
            summary["observation_ended_at_s"] = now()
            summary["end_buffers"] = buffer_snapshot()
            mark_cleanup()
            for task in tasks[1::2]:
                task.cancel()
            await asyncio.gather(*tasks[1::2], return_exceptions=True)
            for role in sockets:
                await send(role, {"type": "session.close"})
            await asyncio.wait_for(asyncio.gather(*tasks[::2]), timeout=30)
            summary["status"] = "completed"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        mark_cleanup()
        resume.set()
        if held_lock is not None:
            held_lock.release()
        for task in tasks:
            task.cancel()
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        summary["errors"].extend(
            {"type": type(result).__name__, "message": str(result)}
            for result in task_results
            if isinstance(result, Exception)
        )
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=15)
        except Exception as exc:
            summary["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        cancel_at = summary.get("cancel_submitted_at_s", float("inf"))
        terminal_at = summary.get("cancelled_terminal_at_s", float("inf"))
        intervention_at = summary.get("intervention_at_s", float("inf"))
        old = responses.get("slow")
        summary["response_ids"] = dict(responses)
        for label, source in (("received", events), ("asgi_accepted", server_events)):
            old_audio = [
                e
                for e in source
                if e["role"] == "slow" and e["response_id"] == old and e["audio_bytes"]
            ]
            summary[f"old_audio_{label}_after_cancel_before_terminal"] = sum(
                e["audio_bytes"]
                for e in old_audio
                if cancel_at <= e["at_s"] < terminal_at
            )
            summary[f"old_audio_{label}_after_terminal"] = sum(
                e["audio_bytes"] for e in old_audio if e["at_s"] >= terminal_at
            )
        for role in first:
            own = [e for e in events if e["role"] == role]
            initial = [e for e in own if e["response_id"] == responses.get(role)]
            seq = [
                e["server_event_seq"]
                for e in own
                if isinstance(e["server_event_seq"], int)
            ]
            terminals = {}
            violations = []
            for event in own:
                response_id = event["response_id"]
                if event["type"] == "response.done":
                    terminals[response_id] = event
                if event["audio_bytes"] and response_id in terminals:
                    violations.append(event)
            summary[role] = {
                "audio_at_s": [e["at_s"] for e in own if e["audio_bytes"]],
                "audio_during_pause": sum(
                    bool(e["audio_bytes"])
                    for e in own
                    if summary.get("pause_window_started_at_s", float("inf"))
                    <= e["at_s"]
                    < intervention_at
                ),
                "audio_after_cancel": sum(
                    bool(e["audio_bytes"]) for e in own if e["at_s"] >= cancel_at
                ),
                "initial_response_audio_during_pause": sum(
                    bool(e["audio_bytes"])
                    for e in initial
                    if summary.get("pause_window_started_at_s", float("inf"))
                    <= e["at_s"]
                    < intervention_at
                ),
                "initial_response_audio_after_cancel": sum(
                    bool(e["audio_bytes"]) for e in initial if e["at_s"] >= cancel_at
                ),
                "initial_response_audio_asgi_after_cancel": sum(
                    bool(e["audio_bytes"])
                    for e in server_events
                    if e["role"] == role
                    and e["response_id"] == responses.get(role)
                    and e["at_s"] >= cancel_at
                ),
                "initial_response_audio_after_cancelled_terminal": sum(
                    bool(e["audio_bytes"]) for e in initial if e["at_s"] >= terminal_at
                ),
                "initial_response_audio_asgi_after_cancelled_terminal": sum(
                    bool(e["audio_bytes"])
                    for e in server_events
                    if e["role"] == role
                    and e["response_id"] == responses.get(role)
                    and e["at_s"] >= terminal_at
                ),
                "initial_response_completed": any(
                    e["type"] == "response.done" and e["response_status"] == "completed"
                    for e in initial
                ),
                "other_response_audio_after_cancel": sum(
                    bool(e["audio_bytes"])
                    for e in own
                    if e["response_id"] != responses.get(role)
                    and e["at_s"] >= cancel_at
                ),
                "sequence_strictly_increasing": all(
                    a < b for a, b in zip(seq, seq[1:])
                ),
                "sequence_contiguous": all(b == a + 1 for a, b in zip(seq, seq[1:])),
                "normal_completions": sum(
                    e["response_status"] == "completed"
                    for e in own
                    if e["type"] == "response.done"
                ),
                "audio_after_own_terminal": violations,
            }
        summary["checks"] = {
            "both_sessions_produced_audio": all(
                first[role].is_set() and responses.get(role) for role in first
            ),
            "slow_cancellation_accepted": cancelled.is_set(),
            "normal_initial_response_audio_during_pause": summary["normal"][
                "initial_response_audio_during_pause"
            ]
            > 0,
            "normal_initial_response_audio_after_cancel": summary["normal"][
                "initial_response_audio_after_cancel"
            ]
            > 0,
            "normal_initial_response_audio_asgi_after_cancel": (
                summary["normal"]["initial_response_audio_asgi_after_cancel"] > 0
            ),
            "normal_initial_response_audio_after_cancelled_terminal": (
                summary["normal"]["initial_response_audio_after_cancelled_terminal"] > 0
            ),
            "normal_initial_response_audio_asgi_after_cancelled_terminal": (
                summary["normal"][
                    "initial_response_audio_asgi_after_cancelled_terminal"
                ]
                > 0
            ),
            "normal_initial_response_completed": summary["normal"][
                "initial_response_completed"
            ],
            "sequences_valid": all(
                summary[role]["sequence_strictly_increasing"]
                and summary[role]["sequence_contiguous"]
                for role in first
            ),
            "no_audio_after_own_terminal": all(
                not summary[role]["audio_after_own_terminal"] for role in first
            ),
            "no_errors_or_unexpected_closes": not summary["errors"]
            and not summary["unexpected_closes"],
        }
        if args.cancel_slow == "omit":
            for key in list(summary["checks"]):
                if "cancel" in key:
                    del summary["checks"][key]
            # Omitted cancellation has no stale/after-cancel measurements.
            for key in list(summary):
                if key.startswith("old_audio_"):
                    del summary[key]
            for role in first:
                for key in list(summary[role]):
                    if "after_cancel" in key:
                        del summary[role][key]
            for label, source in (
                ("received", events),
                ("asgi_accepted", server_events),
            ):
                summary["checks"][
                    f"normal_initial_response_audio_{label}_after_release"
                ] = any(
                    e["role"] == "normal"
                    and e["response_id"] == responses.get("normal")
                    and e["audio_bytes"]
                    and e["at_s"] >= summary.get("resume_reads_at_s", float("inf"))
                    for e in source
                )
        # Delivery and natural completion are different observations. Preserve
        # full mode's original deadline and verdict; delivery mode closes after
        # a fixed window and never calls that an observed natural completion.
        window_end = summary.get("delivery_window_ended_at_s")
        window_start = summary.get("delivery_window_started_at_s", float("inf"))
        progress = {}
        for label, source in (("received", events), ("asgi_accepted", server_events)):
            normal_events = [
                event
                for event in source
                if event["role"] == "normal"
                and event["response_id"] == responses.get("normal")
                and window_end is not None
                and window_start <= event["at_s"] <= window_end
            ]
            normal_ending = next(
                (e for e in normal_events if e["type"] == "response.done"), None
            )
            bins = [
                sum(
                    bool(e["audio_bytes"])
                    for e in normal_events
                    if window_start + args.observe_after_s * index / 3
                    <= e["at_s"]
                    < window_start + args.observe_after_s * (index + 1) / 3
                )
                for index in range(3)
            ]
            progress[label] = {
                "audio_per_third": bins,
                "natural_completion": (
                    normal_ending["response_status"]
                    if normal_ending is not None
                    else "not_observed_within_window"
                ),
                "continued_or_completed": window_end is not None
                and all(
                    count > 0
                    or (
                        normal_ending is not None
                        and normal_ending["response_status"] == "completed"
                        and normal_ending["at_s"]
                        < window_start + args.observe_after_s * (index + 1) / 3
                    )
                    for index, count in enumerate(bins)
                )
                and (
                    normal_ending is None
                    or normal_ending["response_status"] == "completed"
                ),
            }
        summary["delivery_window"] = progress
        if args.completion_mode == "delivery":
            summary["checks"].pop("normal_initial_response_completed")
            summary["checks"]["normal_progress_throughout_delivery_window"] = all(
                value["continued_or_completed"] for value in progress.values()
            )
        if not all(summary["checks"].values()):
            summary["status"] = "failed"
        trace.close()
        args.result_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in (
            ("summary.json", summary),
            ("events.json", events),
            ("server_events.json", server_events),
            ("termination_trace.json", trace.snapshot(roles)),
        ):
            (args.result_dir / filename).write_text(json.dumps(content, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--deploy-config", type=Path, default=Path("vllm_omni/deploy/minicpmo_4_5.yaml")
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        help="Reference WAV; defaults to the local model's assets/HT_ref_audio.wav",
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("network", "held"), required=True)
    parser.add_argument(
        "--completion-mode",
        choices=("full", "delivery"),
        default="full",
        help="full retains the natural-completion deadline; delivery observes a fixed window then closes",
    )
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--cancel-slow", choices=("request", "omit"), default="request")
    parser.add_argument("--trace-termination", action="store_true")
    parser.add_argument("--text-repetitions", type=int, default=12)
    parser.add_argument("--pause-before-s", type=float, default=6)
    parser.add_argument("--observe-after-s", type=float, default=30)
    args = parser.parse_args()
    if args.pause_before_s <= 0 or args.observe_after_s <= 0:
        parser.error("--pause-before-s and --observe-after-s must be positive")
    if args.text_repetitions <= 0:
        parser.error("--text-repetitions must be positive")
    from vllm_omni.entrypoints.duplex_omni import DuplexOmni

    omni = DuplexOmni(
        model=args.model,
        trust_remote_code=True,
        # 8b028c9 lost its Path import in the explicit-string normalization
        # branch. Pass the same file as a supported Path, without patching
        # either version's production source or changing its stage settings.
        deploy_config=args.deploy_config,
    )
    try:
        summary = asyncio.run(observe(omni, args))
        print(json.dumps(summary, indent=2))
        if summary["status"] != "completed":
            raise SystemExit(1)
    finally:
        omni.shutdown()


if __name__ == "__main__":
    main()
