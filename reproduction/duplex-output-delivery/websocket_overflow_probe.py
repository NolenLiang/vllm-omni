# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real MiniCPM-o overflow through the production handler and loopback WebSockets.

Like websocket_delivery_probe.py, hold one production outbound lock before
sequence allocation. Unlike its cancellation experiment, submit no cancellation:
the output budget alone must close the slow session. This deliberately installed
server hold is not a measurement of physical network delay or playback.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import traceback
from contextlib import AsyncExitStack
from pathlib import Path


async def observe(omni, args, summary, events, server_events):
    import uvicorn
    import websockets
    from fastapi import FastAPI, WebSocket

    from vllm_omni.clients.duplex import SessionConfig
    from vllm_omni.engine.duplex.events import AudioDelta
    from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler

    started = time.monotonic()
    roles, handles, identities, responses, sockets, send_locks = {}, {}, {}, {}, {}, {}
    first = {role: asyncio.Event() for role in ("slow", "normal", "replacement")}
    opened = {role: asyncio.Event() for role in first}
    closed = {role: asyncio.Event() for role in first}
    held, normal_after_close = asyncio.Event(), asyncio.Event()
    tasks, feeders = [], []
    held_lock = held_event = None
    closing = False
    normal_after_count = 0

    def now():
        return time.monotonic() - started

    def compact(payload, role, phase):
        response = payload.get("response", {})
        kind = payload.get("type")
        delta = payload.get("delta", "")
        metadata = response.get("metadata", {})
        return {
            "at_s": now(),
            "role": role,
            "phase": phase,
            "type": kind,
            "event_id": payload.get("event_id"),
            "server_event_seq": payload.get("server_event_seq"),
            "response_id": payload.get("response_id") or response.get("id"),
            "response_status": response.get("status"),
            "status_details": response.get("status_details"),
            "committed": metadata.get("committed"),
            "reason": payload.get("reason"),
            "epoch": identities.get(payload.get("event_id"), {}).get("epoch"),
            "audio_bytes": len(base64.b64decode(delta))
            if kind == AudioDelta.wire_type and delta
            else 0,
            **({"error": payload.get("error")} if kind == "error" else {}),
        }

    class ObservedHandler(OmniDuplexSessionHandler):
        async def _open(self, websocket, envelope, session_payload, send_json):
            attachment = await super()._open(
                websocket, envelope, session_payload, send_json
            )
            if attachment is not None:
                role = websocket.query_params["role"]
                handles[role] = attachment.handle
                roles[attachment.handle.session_id] = role
            return attachment

        async def _send_event(self, session_id, event, **kwargs):
            nonlocal held_lock, held_event
            identities[event.event_id] = {
                "epoch": event.epoch,
                "session_id": session_id,
            }
            if roles.get(session_id) == "slow" and isinstance(event, AudioDelta):
                summary["slow_audio_attempts"] = (
                    summary.get("slow_audio_attempts", 0) + 1
                )
                if summary["slow_audio_attempts"] == 2:
                    held_lock = self._attachment_registry._sessions[
                        session_id
                    ].outbound_lock
                    await held_lock.acquire()
                    held_event = event
                    summary.update(
                        held_at_s=now(),
                        held_event_id=event.event_id,
                        held_response_id=event.response_id,
                    )
                    held.set()
            await super()._send_event(session_id, event, **kwargs)
            if event is held_event:
                state = self._attachment_registry._sessions.get(session_id)
                summary["held_event_in_journal"] = (
                    any(
                        entry.payload.get("event_id") == event.event_id
                        for entry in state.journal._entries
                    )
                    if state is not None
                    else None
                )
                summary["journal_observed_at_s"] = now()

    handler = ObservedHandler(duplex_omni=omni)
    app = FastAPI()

    async def route(websocket):
        await handler.handle_realtime_session(websocket)

    route.__annotations__["websocket"] = WebSocket
    app.websocket("/v1/realtime")(route)

    async def observed_app(scope, receive, send):
        query = scope.get("query_string", b"").decode()
        role = next((role for role in first if f"role={role}" in query), "unknown")

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
            if role in handles and handles[role].closed:
                return
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
        nonlocal normal_after_count
        try:
            async for raw in sockets[role]:
                entry = compact(json.loads(raw), role, "client_receive")
                events.append(entry)
                if entry["type"] == "session.created":
                    opened[role].set()
                if entry["type"] == "error" and not (
                    role == "slow"
                    and entry["error"].get("code") == "output_backpressure"
                ):
                    raise RuntimeError(f"Unexpected error: {entry}")
                if entry["audio_bytes"]:
                    responses.setdefault(role, entry["response_id"])
                    first[role].set()
                    if (
                        role == "normal"
                        and closed["slow"].is_set()
                        and entry["response_id"] == responses[role]
                    ):
                        normal_after_count += 1
                        if normal_after_count >= 3:
                            normal_after_close.set()
                if entry["type"] in {"session.closed", "session.expired"}:
                    if not closing and not (
                        role == "slow" and entry["reason"] == "output_backpressure"
                    ):
                        raise RuntimeError(f"Unexpected closure: {entry}")
                    summary[f"{role}_client_closed_at_s"] = entry["at_s"]
                    closed[role].set()
        except websockets.exceptions.ConnectionClosed:
            if not closing and not closed[role].is_set():
                raise
        if not closing and not closed[role].is_set():
            raise RuntimeError(f"{role} WebSocket ended without a terminal event")

    async def open_socket(stack, role):
        uri = f"ws://127.0.0.1:{args.port}/v1/realtime?duplex=1&autostart=0&role={role}"
        sockets[role] = await stack.enter_async_context(
            websockets.connect(
                uri, max_size=8 * 1024 * 1024, max_queue=1, ping_interval=None
            )
        )
        send_locks[role] = asyncio.Lock()
        story = (
            "Please read this story aloud. A scientist walked through a quiet garden on a clear morning. "
            "She stopped to examine each flower and carefully record its colour, shape, and scent in her notebook. "
            "After an hour she sat beside the pond and began writing a letter to her friend. "
        )
        config = SessionConfig(
            instructions="Read the supplied text aloud in English, exactly as written.",
            overlap_policy="listen_only",
            playback_commit_policy="ack_only",
            extra_body={
                "auto_response": True,
                "duplex_initial_user_text": story * 24,
                "force_listen_count": 0,
            },
        ).to_session_payload(model=args.model)
        reference = (Path(args.model) / "assets" / "HT_ref_audio.wav").read_bytes()
        config["ref_audio"] = (
            "data:audio/wav;base64," + base64.b64encode(reference).decode()
        )
        await send(role, {"type": "session.update", "session": config})
        tasks.append(asyncio.create_task(receive(role)))
        feeders.append(asyncio.create_task(feed(role)))

    try:
        async with asyncio.timeout(240), AsyncExitStack() as stack:
            while not server.started:
                if server_task.done():
                    await server_task
                    raise RuntimeError("WebSocket server did not start")
                await asyncio.sleep(0.05)
            await open_socket(stack, "slow")
            await asyncio.wait_for(
                asyncio.gather(first["slow"].wait(), held.wait()),
                timeout=60,
            )
            # Keep the normal response young when the slow budget expires.
            # Simultaneous starts previously completed it before client closure.
            await asyncio.sleep(5)
            summary["normal_input_started_at_s"] = now()
            await open_socket(stack, "normal")
            await asyncio.wait_for(first["normal"].wait(), timeout=60)
            slow_handle = handles["slow"]
            if slow_handle.closed or handles["normal"].closed:
                raise RuntimeError(
                    "Both sessions must be open when the output hold starts"
                )
            summary["hold_observed_at_s"] = now()
            # No cancel, close or release: only the configured output budget can terminate this session.
            reason = await asyncio.wait_for(slow_handle.wait_closed(), timeout=40)
            summary["slow_engine_closed_at_s"] = now()
            if (
                reason != "output_backpressure"
                or handles["normal"].closed
                or omni.active_session_count() != 1
            ):
                raise RuntimeError(
                    "Overflow did not close only the held session and release its capacity"
                )
            summary["active_sessions_after_overflow"] = omni.active_session_count()
            with slow_handle.output_guard(held_event) as valid:
                summary["held_event_valid_before_release"] = valid
            summary["release_hold_at_s"] = now()
            held_lock.release()
            held_lock = None
            await asyncio.wait_for(closed["slow"].wait(), timeout=30)
            await asyncio.wait_for(normal_after_close.wait(), timeout=20)
            await open_socket(stack, "replacement")
            await asyncio.wait_for(first["replacement"].wait(), timeout=60)
            summary["active_sessions_after_replacement"] = omni.active_session_count()
            summary["observation_ended_at_s"] = now()
            closing = True
            for task in feeders:
                task.cancel()
            await asyncio.gather(*feeders, return_exceptions=True)
            for role in ("normal", "replacement"):
                await send(role, {"type": "session.close"})
            await asyncio.wait_for(
                asyncio.gather(*(closed[role].wait() for role in first)), timeout=30
            )
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)
            summary["status"] = "completed"
    finally:
        closing = True
        if held_lock is not None:
            held_lock.release()
        for task in tasks + feeders:
            task.cancel()
        results = await asyncio.gather(*tasks, *feeders, return_exceptions=True)
        summary["background_errors"] = [
            str(result) for result in results if isinstance(result, Exception)
        ]
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=15)
        summary["response_ids"] = responses
        summary["session_ids"] = {
            role: handle.session_id for role, handle in handles.items()
        }

    slow = [event for event in events if event["role"] == "slow"]
    terminals = [
        event
        for event in slow
        if event["type"] in {"error", "response.done", "session.closed"}
    ]
    normal_audio = [
        event
        for event in events
        if event["role"] == "normal"
        and event["response_id"] == responses.get("normal")
        and event["audio_bytes"]
    ]
    summary["normal_same_response_audio_while_held"] = sum(
        summary["held_at_s"] <= event["at_s"] < summary["slow_engine_closed_at_s"]
        for event in normal_audio
    )
    summary["normal_same_response_audio_after_client_close"] = sum(
        summary["slow_client_closed_at_s"]
        <= event["at_s"]
        < summary["observation_ended_at_s"]
        for event in normal_audio
    )
    sequences = [
        [
            event["server_event_seq"]
            for event in events
            if event["role"] == role and isinstance(event["server_event_seq"], int)
        ]
        for role in first
    ]
    endings = {}
    late_audio = []
    for event in events:
        key = (event["role"], event["response_id"])
        if event["type"] == "response.done":
            endings[key] = event
        if event["audio_bytes"] and key in endings:
            late_audio.append(event)
    summary["checks"] = {
        "overflow_terminal_order": [event["type"] for event in terminals]
        == ["error", "response.done", "session.closed"],
        "one_failed_original_response": len(terminals) == 3
        and terminals[1]["response_id"] == responses["slow"]
        and terminals[1]["response_status"] == "failed"
        and terminals[1]["committed"] is False,
        "held_audio_invalid_before_release": summary.get(
            "held_event_valid_before_release"
        )
        is False,
        "held_audio_not_journaled": summary.get("held_event_in_journal") is False,
        "held_audio_not_sent_or_received": not any(
            event["event_id"] == summary["held_event_id"]
            for event in events + server_events
        ),
        "other_same_response_across_close": summary[
            "normal_same_response_audio_while_held"
        ]
        > 0
        and summary["normal_same_response_audio_after_client_close"] >= 3,
        "replacement_actual_distinct_response": len(responses)
        == len(set(responses.values()))
        == 3
        and any(
            event["role"] == "replacement" and event["audio_bytes"] > 0
            for event in events
        )
        and summary["active_sessions_after_replacement"] == 2,
        "sequences_contiguous": all(
            all(b == a + 1 for a, b in zip(seq, seq[1:])) for seq in sequences
        ),
        "no_audio_after_own_ending": not late_audio,
        "no_background_errors": not summary["background_errors"],
    }
    if not all(summary["checks"].values()):
        raise RuntimeError(f"Overflow requirements failed: {summary['checks']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-archive", help="Exact source archive used by the launcher")
    parser.add_argument(
        "--deploy-config", type=Path, default=Path("vllm_omni/deploy/minicpmo_4_5.yaml")
    )
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    overlay = args.result_dir.resolve() / "overflow_deploy.yaml"
    overlay.write_text(
        f"base_config: {json.dumps(str(args.deploy_config.resolve()))}\n"
        "duplex_session:\n  max_sessions: 2\n"
        "  max_pending_output_bytes_per_session: 786432\n"
        "  max_pending_output_events_per_session: 512\n"
    )
    summary = {
        "status": "running",
        "source_archive": args.source_archive,
        "scope": "real MiniCPM-o; production handler over loopback WebSockets, not complete OpenAI app",
        "hold": "instrumented production outbound lock before sequence allocation; not physical network delay",
        "close_trigger": "output budget alone; no cancellation request",
        "max_sessions": 2,
        "max_pending_output_bytes_per_session": 786432,
        "max_pending_output_events_per_session": 512,
        "overlap_policy": "listen_only",
        "normal_input_start_delay_after_hold_s": 5,
        "limit": "Normal and replacement sessions are explicitly closed after recovery audio, not natural completion.",
    }
    events, server_events, omni = [], [], None
    try:
        from vllm_omni.entrypoints.duplex_omni import DuplexOmni

        omni = DuplexOmni(
            model=args.model, trust_remote_code=True, deploy_config=overlay
        )
        asyncio.run(observe(omni, args, summary, events, server_events))
    except BaseException as exc:
        summary.update(
            status="failed",
            failure={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        if omni is not None:
            try:
                omni.shutdown()
            except Exception as exc:
                summary.update(status="failed", shutdown_error=str(exc))
        for filename, content in (
            ("summary.json", summary),
            ("events.json", events),
            ("server_events.json", server_events),
        ):
            (args.result_dir / filename).write_text(json.dumps(content, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
