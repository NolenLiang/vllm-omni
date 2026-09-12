# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Observe two real MiniCPM-o sessions sharing one DuplexOmni engine.

Only the slow session's output consumption pauses. Both input feeders and the
other session's output consumption continue. This is not a WebSocket or
physical-playback measurement, and the deliberate pauses are not cancel latency.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path


async def observe(
    omni, model: str, result_dir: Path, *, pause_before_s: float = 12
) -> dict:
    from vllm_omni.clients.duplex import SessionConfig
    from vllm_omni.engine.duplex.events import AudioDelta

    config = SessionConfig(
        instructions="Read the supplied text aloud in English, exactly as written.",
        overlap_policy="listen_only",
        playback_commit_policy="ack_only",
        extra_body={
            "auto_response": True,
            "duplex_initial_user_text": (
                "Please read this story aloud. A scientist walked through a quiet garden "
                "on a clear morning. She stopped to examine each flower and carefully "
                "record its colour, shape, and scent in her notebook. After an hour she "
                "sat beside the pond and began writing a letter to her friend. "
            )
            * 24,
            "force_listen_count": 0,
        },
    )
    reference_audio = Path(model) / "assets" / "HT_ref_audio.wav"
    encoded_reference = base64.b64encode(reference_audio.read_bytes()).decode("ascii")
    handles = {}
    streams = {}
    tasks = []
    events = []
    started = time.monotonic()
    closing = False
    summary = {
        "scope": "two real MiniCPM-o sessions, one DuplexOmni engine; no WebSocket claim",
        "status": "running",
        "deliberate_pause_before_cancel_s": pause_before_s,
        "deliberate_pause_after_cancel_s": 2,
        "observe_after_cancel_terminal_s": 3,
        "note": (
            "Cancellation submission acknowledges enqueue only. The slow reader deliberately "
            "waits two more seconds, so terminal observation time is not engine cancel latency. "
            "A normal response.done does not establish accepted cancellation."
        ),
    }

    def elapsed():
        return time.monotonic() - started

    def record(label, event):
        entry = {
            "elapsed_s": elapsed(),
            "session": label,
            "type": event.type,
            "response_id": event.response_id,
            "epoch": event.epoch,
            "response_status": getattr(event, "status", None),
            "audio_bytes": len(event.audio or b""),
        }
        if isinstance(event, AudioDelta):
            entry.update(format=event.format, sample_rate_hz=event.sample_rate_hz)
        if event.type == "error":
            entry["error"] = event.to_realtime()
        events.append(entry)
        if event.type == "error":
            raise RuntimeError(f"{label}: {entry['error']}")
        if event.is_terminal and not closing:
            raise RuntimeError(f"{label} closed unexpectedly: {entry}")
        return entry

    def queue_state(handle):
        # The source baseline has asyncio.Queue; the patch has DuplexOutputBuffer.
        outbox = handle._outbox
        count = getattr(outbox, "pending_events", None)
        return {
            "pending_events": count if count is not None else outbox.qsize(),
            "pending_bytes": getattr(outbox, "pending_bytes", None),
        }

    async def feed_silence(handle):
        for _ in range(300):
            await handle.append_audio(
                bytes(6400),
                format="pcm16",
                sample_rate_hz=16000,
                is_speech=False,
                duration_ms=200,
            )
            await asyncio.sleep(0.2)

    async def consume_normal():
        async for event in streams["normal"]:
            record("normal", event)
        if not closing:
            raise RuntimeError("normal event stream ended before requested closure")

    try:
        for label in ("normal", "slow"):
            payload = config.to_session_payload(model=model)
            payload["ref_audio"] = "data:audio/wav;base64," + encoded_reference
            handles[label] = await omni.open_session(payload, timeout=120)
            streams[label] = handles[label].events()
        summary["session_ids"] = {
            label: handle.session_id for label, handle in handles.items()
        }
        started = time.monotonic()

        # TaskGroup surfaces a feeder/normal-consumer failure immediately, including
        # while the slow reader sleeps, instead of hiding it behind a timeout.
        async with asyncio.timeout(240), asyncio.TaskGroup() as group:
            feeders = [
                group.create_task(feed_silence(handle)) for handle in handles.values()
            ]
            normal_reader = group.create_task(consume_normal())
            tasks.extend([*feeders, normal_reader])
            old_response = None
            while old_response is None:
                entry = record("slow", await anext(streams["slow"]))
                if entry["audio_bytes"] and entry["response_id"]:
                    old_response = entry["response_id"]
                    summary["slow_first_audio_at_s"] = entry["elapsed_s"]

            summary["slow_response_id"] = old_response
            summary["slow_pause_started_at_s"] = elapsed()
            await asyncio.sleep(summary["deliberate_pause_before_cancel_s"])
            summary["slow_queue_before_cancel"] = queue_state(handles["slow"])
            summary["cancel_submitted_at_s"] = elapsed()
            await handles["slow"].cancel_response(old_response)
            await asyncio.sleep(summary["deliberate_pause_after_cancel_s"])
            summary["slow_queue_after_cancel"] = queue_state(handles["slow"])
            summary["slow_reads_resumed_at_s"] = elapsed()

            async with asyncio.timeout(60):
                while "cancel_terminal" not in summary:
                    entry = record("slow", await anext(streams["slow"]))
                    if entry["response_id"] == old_response and (
                        entry["type"] == "audio.cancelled"
                        or (
                            entry["type"] == "response.done"
                            and entry["response_status"] == "cancelled"
                        )
                    ):
                        summary["cancel_terminal"] = entry

            # Keep consuming even after the cancelled response's terminal event.
            # Expiry intentionally closes this generator; cleanup uses handle.close.
            try:
                async with asyncio.timeout(summary["observe_after_cancel_terminal_s"]):
                    while True:
                        record("slow", await anext(streams["slow"]))
            except TimeoutError:
                pass
            summary["observation_ended_at_s"] = elapsed()
            summary["queues_at_observation_end"] = {
                label: queue_state(handle) for label, handle in handles.items()
            }

            closing = True
            for feeder in feeders:
                feeder.cancel()
            await asyncio.gather(*feeders, return_exceptions=True)
            await asyncio.gather(
                *(handle.close(timeout=30) for handle in handles.values())
            )
            await asyncio.wait_for(normal_reader, timeout=10)

        summary["status"] = "completed"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        closing = True
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        cleanup_errors = []
        for label, handle in handles.items():
            try:
                if label in streams:
                    await streams[label].aclose()
                await handle.close(timeout=30)
                await asyncio.wait_for(handle.wait_closed(), timeout=10)
            except BaseException as exc:
                cleanup_errors.append(
                    {"session": label, "type": type(exc).__name__, "message": str(exc)}
                )
        summary["closed_sessions"] = {
            label: {"closed": handle.closed, "reason": handle.close_reason}
            for label, handle in handles.items()
        }
        if cleanup_errors:
            summary.update(status="failed", cleanup_errors=cleanup_errors)

        cancel_at = summary.get("cancel_submitted_at_s", float("inf"))
        terminal_at = summary.get("cancel_terminal", {}).get("elapsed_s", float("inf"))
        paused_at = summary.get("slow_pause_started_at_s", float("inf"))
        resumed_at = summary.get("slow_reads_resumed_at_s", float("-inf"))
        ended_at = summary.get("observation_ended_at_s", float("-inf"))
        audio = [entry for entry in events if entry["audio_bytes"]]
        old_audio = [
            entry
            for entry in audio
            if entry["session"] == "slow"
            and entry["response_id"] == summary.get("slow_response_id")
        ]
        groups = {
            "slow_old_audio_after_cancel_submission": [
                entry for entry in old_audio if entry["elapsed_s"] >= cancel_at
            ],
            "slow_old_audio_after_cancel_terminal": [
                entry for entry in old_audio if entry["elapsed_s"] > terminal_at
            ],
            "normal_audio_while_slow_reader_paused": [
                entry
                for entry in audio
                if entry["session"] == "normal"
                and paused_at <= entry["elapsed_s"] < resumed_at
            ],
            "normal_audio_after_slow_cancel_terminal": [
                entry
                for entry in audio
                if entry["session"] == "normal"
                and terminal_at <= entry["elapsed_s"] <= ended_at
            ],
        }
        summary["audio_observations"] = {
            name: {
                "events": len(entries),
                "bytes": sum(entry["audio_bytes"] for entry in entries),
                "first_at_s": entries[0]["elapsed_s"] if entries else None,
                "last_at_s": entries[-1]["elapsed_s"] if entries else None,
            }
            for name, entries in groups.items()
        }
        normal_times = [
            entry["elapsed_s"] for entry in audio if entry["session"] == "normal"
        ]
        summary["normal_max_observed_audio_gap_s"] = max(
            (later - earlier for earlier, later in zip(normal_times, normal_times[1:])),
            default=None,
        )
        summary["terminal_cancelled"] = "cancel_terminal" in summary
        summary["normal_session_continued"] = bool(
            groups["normal_audio_while_slow_reader_paused"]
            and groups["normal_audio_after_slow_cancel_terminal"]
        )
        if not summary["terminal_cancelled"] or not summary["normal_session_continued"]:
            summary["status"] = "failed"
            summary["insufficient_observation"] = (
                "Requires accepted cancellation and actual normal-session audio both during "
                "the slow reader's pause and after the slow response's cancelled terminal."
            )
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "events.json").write_text(json.dumps(events, indent=2))
        (result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--pause-before-s", type=float, default=12)
    parser.add_argument(
        "--deploy-config", type=Path, default=Path("vllm_omni/deploy/minicpmo_4_5.yaml")
    )
    args = parser.parse_args()
    if args.pause_before_s <= 0:
        parser.error("--pause-before-s must be positive")
    from vllm_omni.entrypoints.duplex_omni import DuplexOmni

    omni = DuplexOmni(
        model=args.model, trust_remote_code=True, deploy_config=args.deploy_config
    )
    try:
        summary = asyncio.run(
            observe(
                omni, args.model, args.result_dir, pause_before_s=args.pause_before_s
            )
        )
        print(json.dumps(summary, indent=2))
        if summary["status"] != "completed":
            raise SystemExit(1)
    finally:
        omni.shutdown()


if __name__ == "__main__":
    main()
