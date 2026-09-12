# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real MiniCPM-o: a stopped reader closes only its session on output overflow."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import traceback
from pathlib import Path


async def observe(omni, args, summary, events):
    from vllm_omni.clients.duplex import SessionConfig
    from vllm_omni.engine.duplex.events import AudioDelta, ErrorEvent, ResponseDone, SessionClosed
    from vllm_omni.engine.duplex.messages import DuplexSessionError

    config = SessionConfig(
        instructions="Read the supplied text aloud in English, exactly as written.",
        overlap_policy="listen_only", playback_commit_policy="ack_only",
        extra_body={
            "auto_response": True, "force_listen_count": 0,
            "duplex_initial_user_text": (
                "Please read this story aloud. A scientist walked through a quiet garden "
                "on a clear morning. She stopped to examine each flower and carefully "
                "record its colour, shape, and scent in her notebook. After an hour she "
                "sat beside the pond and began writing a letter to her friend. "
            ) * 24,
        },
    ).to_session_payload(model=args.model)
    reference = (Path(args.model) / "assets" / "HT_ref_audio.wav").read_bytes()
    config["ref_audio"] = "data:audio/wav;base64," + base64.b64encode(reference).decode()
    handles, streams, responses, tasks = {}, {}, {}, []
    ended_responses = set()
    normal_first, normal_after_close = asyncio.Event(), asyncio.Event()
    started, closing = time.monotonic(), False

    def elapsed():
        return time.monotonic() - started

    def record(role, event):
        audio = isinstance(event, AudioDelta)
        entry = {
            "at_s": elapsed(), "role": role, "session_id": event.session_id,
            "type": event.type, "response_id": event.response_id, "epoch": event.epoch,
            "audio_bytes": len(event.audio or b"") if audio else 0,
        }
        if audio:
            if (role, event.response_id) in ended_responses:
                raise RuntimeError(f"Audio followed its response terminal: {entry}")
            entry.update(format=event.format, sample_rate_hz=event.sample_rate_hz)
            if entry["audio_bytes"] and event.response_id:
                responses.setdefault(role, event.response_id)
                if role == "normal":
                    normal_first.set()
                    if event.response_id == responses[role] and entry["at_s"] >= summary.get("slow_closed_at_s", float("inf")):
                        normal_after_close.set()
        if isinstance(event, ResponseDone):
            entry["status"] = event.status
            ended_responses.add((role, event.response_id))
        if isinstance(event, ErrorEvent):
            entry["error"] = event.to_realtime()
            entry["code"] = event.code
        if isinstance(event, SessionClosed):
            entry["reason"] = event.reason
        events.append(entry)
        if isinstance(event, ErrorEvent) and not (role == "slow" and event.code == "output_backpressure"):
            raise RuntimeError(f"Unexpected {role} error: {entry}")
        if isinstance(event, ResponseDone) and event.status != "completed" and not (role == "slow" and event.status == "failed"):
            raise RuntimeError(f"Unexpected {role} response ending: {entry}")
        if event.is_terminal and not closing and not (role == "slow" and entry.get("reason") == "output_backpressure"):
            raise RuntimeError(f"Unexpected {role} closure: {entry}")
        return entry

    async def feed(role):
        for _ in range(600):
            try:
                await handles[role].append_audio(
                    bytes(6400), format="pcm16", sample_rate_hz=16000,
                    is_speech=False, duration_ms=200,
                )
            except DuplexSessionError as exc:
                if role == "slow" and exc.code == "session_closed" and handles[role].close_reason == "output_backpressure":
                    return
                raise
            await asyncio.sleep(0.2)

    async def consume_normal():
        async for event in streams["normal"]:
            record("normal", event)
        if not closing:
            raise RuntimeError("Normal stream ended before requested cleanup")

    async def first_audio(role):
        async with asyncio.timeout(60):
            async for event in streams[role]:
                entry = record(role, event)
                if entry["audio_bytes"] and entry["response_id"]:
                    return entry
        raise RuntimeError(f"{role} ended without audio")

    try:
        async with asyncio.timeout(240), asyncio.TaskGroup() as group:
            for role in ("normal", "slow"):
                handles[role] = await omni.open_session(config, timeout=120)
                streams[role] = handles[role].events()
            tasks.append(group.create_task(consume_normal()))
            slow_feeder = group.create_task(feed("slow"))
            tasks.append(slow_feeder)
            summary["slow_first_audio"] = await first_audio("slow")
            if handles["slow"].closed or handles["normal"].closed:
                raise RuntimeError("Both sessions must still be open when slow reading stops")
            summary["slow_reads_stopped_at_s"] = elapsed()
            tasks.append(group.create_task(feed("normal")))
            await asyncio.wait_for(normal_first.wait(), timeout=60)
            # No cancel, close, or further reads: the engine must close autonomously.
            reason = await asyncio.wait_for(handles["slow"].wait_closed(), timeout=30)
            summary["slow_closed_at_s"] = elapsed()
            if reason != "output_backpressure":
                raise RuntimeError(f"Slow session closed for {reason!r}, not output overflow")
            slow_feeder.cancel()
            await asyncio.gather(slow_feeder, return_exceptions=True)
            async for event in streams["slow"]:
                record("slow", event)
            await asyncio.wait_for(normal_after_close.wait(), timeout=15)

            slow = [e for e in events if e["role"] == "slow"]
            errors = [i for i, e in enumerate(slow) if e.get("code") == "output_backpressure"]
            failed = [i for i, e in enumerate(slow) if e.get("status") == "failed" and e["response_id"] == responses["slow"]]
            closed = [i for i, e in enumerate(slow) if e.get("reason") == "output_backpressure"]
            if not errors or not failed or len(closed) != 1 or not errors[0] < failed[0] < closed[0]:
                raise RuntimeError("Missing or out-of-order overflow error, failed response.done, or session.closed")
            if handles["normal"].closed or omni.active_session_count() != 1:
                raise RuntimeError("Overflow must leave exactly the normal session open")
            handles["replacement"] = await omni.open_session(config, timeout=60)
            if omni.active_session_count() != 2 or len({h.session_id for h in handles.values()}) != 3:
                raise RuntimeError("Replacement must occupy the released slot with a distinct session")
            streams["replacement"] = handles["replacement"].events()
            tasks.append(group.create_task(feed("replacement")))
            summary["replacement_first_audio"] = await first_audio("replacement")
            if len(set(responses.values())) != 3:
                raise RuntimeError("Replacement audio must belong to a distinct response")
            summary["active_sessions_after_replacement"] = omni.active_session_count()
            summary["observed_until_s"] = elapsed()
            summary["status"] = "completed"
            closing = True
            for task in tasks:
                task.cancel()
    finally:
        closing = True
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        summary["session_ids"] = {role: handle.session_id for role, handle in handles.items()}
        summary["response_ids"] = responses
        for role, handle in handles.items():
            try:
                await streams[role].aclose()
                await handle.close(reason="probe_cleanup", timeout=30)
                await asyncio.wait_for(handle.wait_closed(), timeout=10)
            except Exception as exc:
                summary.setdefault("cleanup_errors", []).append({"role": role, "error": str(exc)})
        summary["closed_sessions"] = {
            role: {"closed": handle.closed, "reason": handle.close_reason} for role, handle in handles.items()
        }
        normal_audio = [e for e in events if e["role"] == "normal" and e["response_id"] == responses.get("normal") and e["audio_bytes"]]
        summary["normal_same_response_audio_during_pause"] = sum(
            summary.get("slow_reads_stopped_at_s", float("inf")) <= e["at_s"] < summary.get("slow_closed_at_s", float("-inf"))
            for e in normal_audio
        )
        summary["normal_same_response_audio_after_slow_closed"] = sum(
            e["at_s"] >= summary.get("slow_closed_at_s", float("inf")) for e in normal_audio
        )
        if summary.get("cleanup_errors") or not summary["normal_same_response_audio_during_pause"] or not summary["normal_same_response_audio_after_slow_closed"]:
            summary["status"] = "failed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--deploy-config", type=Path, default=Path("vllm_omni/deploy/minicpmo_4_5.yaml"))
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
        "status": "running", "scope": "real MiniCPM-o, direct Python handles; not WebSocket or playback",
        "max_sessions": 2, "max_pending_output_bytes_per_session": 786432,
        "max_pending_output_events_per_session": 512, "slow_close_trigger": "output overflow only",
    }
    events, omni = [], None
    try:
        from vllm_omni.entrypoints.duplex_omni import DuplexOmni

        omni = DuplexOmni(model=args.model, trust_remote_code=True, deploy_config=overlay)
        asyncio.run(observe(omni, args, summary, events))
    except BaseException as exc:
        summary.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    finally:
        if omni is not None:
            try:
                omni.shutdown()
            except Exception as exc:
                summary.update(status="failed", shutdown_error=str(exc))
        (args.result_dir / "events.json").write_text(json.dumps(events, indent=2))
        (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
