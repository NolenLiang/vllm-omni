# Duplex output-delivery reproduction

Reproduction material for [Duplex output delivery, PR #1](https://github.com/NolenLiang/vllm-omni/pull/1). This separate branch publishes the observers without adding experimental infrastructure to the implementation PR. The measured production revisions remain the commits below.

## Scope and prerequisites

All three observers run real MiniCPM-o 4.5. The WebSocket observer uses the production duplex handler in a small FastAPI application, not the complete OpenAI application. It opens two WebSocket sessions in one engine and cancels the first response of the deliberately slow reader. The two direct Python-handle observers cover cancellation and output-limit closure, respectively; neither runs the full `InlineDuplexClient`.

- Baseline: `8b028c9`; candidate: `e3af4db`.
- Local model snapshot: `openbmb/MiniCPM-o-4_5`, pinned revision `503e754`, including its reference WAV.
- Runtime used for the completed observations: official vLLM 0.29.0 ARM64 image, one GB200, and Transformers 5.14.1 with the model's additional speech dependencies. Keep unrelated user-site packages outside the container.
- Use the matching checkout's Omni installation, with FastAPI, Uvicorn and websockets available. Do not mix the two source trees.
- Include the checkout's model dependencies: `step-audio2==1.0.0`, `s3tokenizer==0.3.0`, `torchdiffeq==0.2.5` and their required speech dependencies. Keep the image's PyTorch, vLLM and FlashInfer unchanged.
- Place all three scripts (`websocket_delivery_probe.py`, `inline_two_session_probe.py`, and `inline_overflow_probe.py`) beside both checkouts and the `models/` directory containing `MiniCPM-o-4_5`. Run each command from its checkout root.

Set `CUDA_VISIBLE_DEVICES=0` and `VLLM_WORKER_MULTIPROC_METHOD=spawn` for each process. Ensure the matching checkout is on `PYTHONPATH`; keep the same environment for both revisions.

The cancellation observers pass the unchanged deploy file as `pathlib.Path` to avoid the baseline's missing-`Path` import on the string-path route; neither measured production tree was patched for this workaround. The overflow observer also passes a `Path`, for its generated deployment file containing the session/output-limit overrides described below.

## Paired commands

From the baseline checkout:

```bash
python3 ../websocket_delivery_probe.py \
  --model ../models/MiniCPM-o-4_5 \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --ref-audio ../models/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
  --mode held --pause-before-s 6 --observe-after-s 30 \
  --result-dir ../results/base-held
```

From the candidate checkout, after the baseline process exits:

```bash
python3 ../websocket_delivery_probe.py \
  --model ../models/MiniCPM-o-4_5 \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --ref-audio ../models/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
  --mode held --pause-before-s 6 --observe-after-s 30 \
  --result-dir ../results/candidate-held
```

Keep the model, configuration, script and pause settings identical. The unchanged deploy file places all three stages on the same GPU.
`held` additionally holds the production send lock before the second audio event receives a sequence number. This is deliberate instrumentation.
For a separate comparison without that lock hold, use `--mode network` and distinct result directories; this mode pauses client reads only.

## Interpret the artifacts

The probe writes `summary.json`, `events.json` and `server_events.json`. A completed run is a valid observation, not by itself proof that the candidate fixes the problem.

- Require every `checks` entry to pass: the slow response ends as cancelled, and the **same initial response** of the other session produces client audio and server-send returns after that cancelled terminal, then completes normally.
- Compare `old_audio_received_after_cancel_before_terminal` and `old_audio_asgi_accepted_after_cancel_before_terminal`; inspect the corresponding after-terminal counts too.
- In `held` mode, `held_event_in_journal` is captured immediately after the held send attempt, before later expiry could remove its record. Inspect the event ID and sequence in the event files.
- Require valid event sequences and no unexpected errors, closes, or audio following its own response terminal. New response IDs are not old audio.

Cancellation submission is not the exact instant the engine accepts it. Deliberate pauses and terminal observation time are not engine-stop latency.
A returned ASGI send call establishes server-send acceptance, not network delivery or physical speaker playback. Already-sent audio still requires client-side playback cancellation.
In the completed held-send pair, baseline old audio after cancel submission was eight chunks / 384,000 decoded bytes; candidate old audio was zero. Both other responses produced 101 chunks after the cancelled terminal and completed. These are single runs, not performance results. Both versions reported shared-memory cleanup warnings on shutdown; leak-free cleanup is not established.

## Cancellation through direct Python handles

From the candidate checkout, use `inline_two_session_probe.py`:

```bash
python3 ../inline_two_session_probe.py \
  --model ../models/MiniCPM-o-4_5 \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --pause-before-s 6 \
  --result-dir ../results/candidate-inline-cancel
```

The reference WAV is read from the model's `assets/HT_ref_audio.wav`. The explicit six-second pause matches the completed observation; the script's default is twelve seconds. After submitting cancellation, the slow reader waits two more seconds before resuming, then observes for three seconds after the cancelled ending.

Inspect `summary.json` and `events.json`: require `status: completed`, `terminal_cancelled: true`, and `normal_session_continued: true`. The candidate observation had zero old audio after cancellation submission, nine other-session chunks while the slow reader paused, and three after the cancelled ending. Both sessions were explicitly closed after observation; this does not claim natural response completion. This is a single candidate run, not a new paired baseline comparison, WebSocket measurement, or full `InlineDuplexClient` run. Shared-memory cleanup warnings remained at shutdown.

## Output-limit closure through direct Python handles

From the candidate checkout, use the companion `inline_overflow_probe.py`:

```bash
python3 ../inline_overflow_probe.py \
  --model ../models/MiniCPM-o-4_5 \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --result-dir ../results/candidate-overflow
```

The script inherits the model deployment and overrides only the session limit (two), queued output bytes (768 KiB), and queued event count (512). The production default remains 2 MiB. The reference audio must be present under the local model's `assets/HT_ref_audio.wav`; opening messages include it, so an artificially small limit can fail before generation even starts.

The slow reader stops after its first audio event; no cancel or close is sent to trigger the overflow. Normal input starts afterwards to keep its response active across closure. Require `output_backpressure`, a failed ending for the slow response, then `session.closed`, followed by audio from the same normal response. A new session must occupy the released slot and produce audio with distinct session and response IDs.

The completed candidate run observed nine normal-response chunks during the pause and three after automatic closure. Active sessions returned to two when the replacement produced its first audio. Normal and replacement sessions were explicitly closed afterwards; natural completion of those responses is not claimed. This is a single direct-handle run, not a WebSocket overflow or physical-playback measurement. Shared-memory cleanup warnings remained at shutdown.

AI assistance: Codex helped with the observer, checks and documentation.
