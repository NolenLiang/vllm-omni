# Duplex output-delivery reproduction

Companion material for [PR #7644](https://github.com/vllm-project/vllm-omni/pull/7644).
The four Python files retain the execution content of the frozen GPU runs; only missing license headers were added. They contain observation instrumentation, not production fixes; the held-send and termination observers wrap runtime methods within the probe process.

Baseline: `e78a5d0`. Public candidate: `5592f8c`, whose source tree is identical to the GPU-tested `d0b5699`.
Prepare two clean checkouts at these revisions. Do not change either checkout during a run.
Keep all four scripts together: `websocket_delivery_probe.py` imports the adjacent `termination_trace.py`, even when tracing is disabled.

## Environment

The recorded runs used one GB200 GPU for all three MiniCPM-o stages, the official `vllm/vllm-openai:v0.29.0` ARM64 image, and additional Python dependencies.
This directory is an observation recipe, **not** a complete environment installer or a claim that the image alone can run MiniCPM-o.
Use the same prepared environment for both checkouts, without replacing its CUDA-enabled PyTorch/vLLM stack.
The observers require Python 3.11 or newer (`TaskGroup` and `asyncio.timeout`).

Recorded versions, from the image/addition package lists and successful import preflight:

| Component | Version |
| --- | --- |
| vLLM / PyTorch / FlashInfer | 0.29.0 / 2.13.0+cu130 / 0.6.18 |
| NumPy / protobuf | 2.2.6 / 6.33.6 |
| Effective Transformers / tokenizers | 5.14.1 / 0.22.2 |
| FastAPI / Uvicorn / websockets | 0.136.3 / 0.52.4 / 17.1 |
| step-audio2 / s3tokenizer / torchdiffeq | 1.0.0 / 0.3.0 / 0.2.5 |
| HyperPyYAML / ONNX / ONNX Runtime | 1.2.3 / 1.22.0 / 1.29.0 |
| SoundFile / SciPy / PyAV | 0.14.0 / 1.18.1 / 18.1.0 |

The speech additions must provide `cosyvoice2` and its dependencies. This small version list is not a full dependency lockfile.
The effective Transformers/tokenizers versions came from additions, not the image's bundled versions.
Use a complete local `openbmb/MiniCPM-o-4_5` snapshot at revision `503e754`, including the tokenizer, model code, weights, `assets/token2wav/`, and `assets/HT_ref_audio.wav`.
Model code is loaded with `trust_remote_code=True`; use the intended trusted snapshot.

The commands use each checkout's `vllm_omni/deploy/minicpmo_4_5.yaml` explicitly. Stage 0/1 have `enforce_eager=False`; this is not an all-eager run.
Allow writable result and compilation/cache directories. Do not interpret initialization or compilation duration as cancellation latency.

## Paths and result capture

Replace these absolute paths. `REPRO_DIR` is this directory, not either code checkout.
Leave `EXTRA_RUNTIME_PATHS` empty if the prepared environment has its additions installed; otherwise set it to their colon-separated Python directories in the intended precedence order.

```bash
export REPRO_DIR=/absolute/path/to/reproduction/duplex-output-delivery
export MODEL_DIR=/absolute/path/to/MiniCPM-o-4_5
export RESULT_DIR=/absolute/path/to/new-duplex-results
export BASELINE_DIR=/absolute/path/to/baseline-checkout
export CANDIDATE_DIR=/absolute/path/to/candidate-checkout
export EXTRA_RUNTIME_PATHS=""
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$RESULT_DIR"
test "$(git -C "$BASELINE_DIR" rev-parse --short=7 HEAD)" = e78a5d0
test "$(git -C "$CANDIDATE_DIR" rev-parse --short=7 HEAD)" = 5592f8c
```

This helper selects the checkout in `PYTHONPATH`, uses an absolute deployment path, refuses to overwrite a case directory, and saves the Python exit code including its existing shutdown call.
A zero exit code does not imply warning-free resource cleanup; retain the complete log.

```bash
run_probe() (
  run_source=$1
  run_case=$2
  run_script=$3
  shift 3
  run_result="$RESULT_DIR/$run_case"
  mkdir "$run_result" || exit 1
  cd "$run_source" || exit 1
  export PYTHONPATH="$run_source${EXTRA_RUNTIME_PATHS:+:$EXTRA_RUNTIME_PATHS}"
  set +e
  python3 "$REPRO_DIR/$run_script" \
    --model "$MODEL_DIR" \
    --deploy-config "$run_source/vllm_omni/deploy/minicpmo_4_5.yaml" \
    --result-dir "$run_result" "$@" >"$run_result/model-probe.log" 2>&1
  run_status=$?
  printf 'probe_and_shutdown_status=%s\n' "$run_status" >"$run_result/exit.status"
  exit "$run_status"
)
```

Run sequentially on the same GPU. Concurrent copies require separate GPU resources and non-conflicting WebSocket ports.

## 1. WebSocket cancellation with a slow receiver

```bash
run_probe "$BASELINE_DIR" ws-baseline websocket_delivery_probe.py \
  --mode held --completion-mode full --cancel-slow request \
  --text-repetitions 3 --pause-before-s 6 --observe-after-s 30 \
  --trace-termination --port 18080

run_probe "$CANDIDATE_DIR" ws-candidate websocket_delivery_probe.py \
  --mode held --completion-mode full --cancel-slow request \
  --text-repetitions 3 --pause-before-s 6 --observe-after-s 30 \
  --trace-termination --port 18080
```

`held` pauses client reads and deliberately holds the slow session's production outbound lock before sequencing its second audio event. Both input feeders and the other receiver continue.
This uses the production handler over loopback WebSockets, not the complete OpenAI application, physical network congestion, or browser playback.
The observer waits two additional seconds before releasing the hold after cancellation; that delay is not engine cancellation latency.
After the 30-second observation window, `full` allows up to 120 additional seconds for the normal original answer to finish.

Read `summary.json`, `events.json`, `server_events.json`, and `termination_trace.json` together.
The recorded pair delivered 8 old audio events / 384,000 decoded bytes after cancellation on baseline and zero on candidate. The held event entered the journal on baseline but not candidate.
Both normal original answers produced 14 audio events / 675,840 bytes, including 3 events after the cancellation terminal, and finished naturally. Later answers in the same session are separate.
The baseline can report all observer checks true: those checks record stale audio rather than requiring its absence. Compare the old-audio counters explicitly.

## 2. WebSocket output limit and replacement session

```bash
run_probe "$CANDIDATE_DIR" ws-overflow websocket_overflow_probe.py --port 18080
```

The script creates `overflow_deploy.yaml` inside its result directory, referencing the selected checkout's deployment file and overriding the limit to 786,432 bytes / 512 events and two sessions. Production defaults are unchanged.
It starts the normal input five seconds after holding the slow output. No cancellation or client close triggers the overflow.
The optional `--source-archive` argument is deliberately omitted: it only copies a label into the summary; it does not extract, select, or verify code. The checkout and `PYTHONPATH` above select the implementation.

The recorded run received `error(output_backpressure)`, the original `response.done(status=failed, committed=false)`, and `session.closed` in that order. The held audio was neither journaled nor delivered.
The normal original answer continued across closure, and a distinct replacement session produced actual audio. Normal/replacement sessions are then explicitly closed; their natural completion is not claimed.

## 3. Direct Python handles

```bash
run_probe "$CANDIDATE_DIR" inline-two inline_two_session_probe.py --pause-before-s 6
```

This observes two direct `DuplexOmni` session handles, not the complete `InlineDuplexClient` interface or WebSockets.
It pauses only one handle's consumption, submits cancellation, waits two further seconds, then observes three seconds after the cancellation terminal before explicit cleanup.
The recorded run delivered zero old audio after cancellation; the normal session continued with 9 events during the pause and 3 after the terminal.

## Limits and retained failures

- WebSocket cancellation defaults to 12 text repetitions; the commands explicitly select the measured short-input value of 3. This repeats the supplied story, not the number of independent runs.
- The overflow and inline scripts retain their separate hard-coded 24-repetition inputs. They do not use the short WebSocket input or wait for both answers to finish naturally.
- These are single observations per scenario, not performance, speech-quality, or statistical reliability claims. Sending audio is not evidence of physical playback; data already sent cannot be recalled.
- Earlier 12-repetition WebSocket completion runs timed out on both baseline and candidate versions while audio continued. The successful short-input comparison does not explain or overturn those failures.
- All four current model processes and outer jobs exited successfully, but logs reported shared-memory cleanup warnings: 84 objects for each WebSocket cancellation run, 27 for overflow, and 30 for inline. This is not evidence of leak-free shutdown.
- The helpers observe private implementation boundaries and may need adaptation after these fixed revisions. Comments referring to an older `8b028c9` string-path issue are retained historical notes, not a claim that this revision still has that defect.

AI assistance: Codex helped prepare the observation scripts and this reproduction guide.
