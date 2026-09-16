# Qwen interrupted-answer history in real Realtime sessions

Prepared evidence for [Item E](https://github.com/vllm-project/vllm-omni/issues/7055#issuecomment-5677251141). This is a separate live-session follow-up to the [text-history comparisons](history-rules.md), not a proposed change to the proportional truncation rule. See the [packet overview](README.md).

## Setup and scope

Qwen3-Omni-30B-A3B-Instruct `26291f7`, vLLM-Omni `a3dee6f`, official vLLM 0.29.0 ARM64, two GB200 GPUs; eager execution and prefix caching disabled. Thinker ran on one GPU, Talker and Code2Wav on the other.

We used `/v1/realtime?duplex=1` with the existing non-native Qwen path, text and audio output, `playback_commit_policy=ack_only`, explicit response creation and no VAD. Three independent sessions each ran once. Playback positions were supplied by the client: there was **no physical playback or human listening**.

Experiment-only observers recorded the actual next Chat request and processed engine prompt without rewriting them. All six request/prompt pairs and their decoded tokens matched. For truncation after generation, we also observed the serving stream task returning; a frontend completion event alone was not our criterion. This is not a measurement of GPU-stage termination.

## Observations

| Action | Received audio at action | Supplied playback position | Assistant history actually passed to the next model request |
| --- | --- | --- | --- |
| Cancel after first text, before first audio | None | 0 ms | `We` |
| Cancel while generating | 302,250 PCM bytes / 6.296875 s | 4,000 ms | 114 characters: first sentence plus part of the second |
| Truncate after the stream returned | 586,410 PCM bytes / 12.216875 s | 4,000 ms | 58 characters: first sentence |

The latter two first answers had identical text. The current approximation used different denominators: the audio generated so far at cancellation, versus the completed audio duration at truncation. Both retained prefixes match the existing formula. We did not verify which words were audible at four seconds and do not propose replacing the approximation on this evidence.

All three follow-up generations completed and sessions closed. The overall experiment **failed** its predeclared “zero played audio retains no assistant text” condition; that failure is retained, not retried away. This condition represents a playback-only policy assumption, not an agreed requirement for every text-and-audio application. No protocol error or environment failure explains the retained text.

### Zero audio is not the same as no delivered text

The client had received text `We`, but no audio. At cancellation, generated, sent, played and committed audio durations were all zero; both the playback acknowledgement and cancellation preceded the next user input. The actual history and engine input nevertheless included assistant `We`.

The [pinned-source rule](https://github.com/vllm-project/vllm-omni/blob/a3dee6f/vllm_omni/entrypoints/duplex/protocol.py#L1578) returns all text when the audio duration is zero, before applying the zero-confirmation branch. This explains the result, but it also serves text-only responses. Moving the zero-confirmation check first without distinguishing output modes would risk deleting legitimate text history.

The next question was `Where and at what time did you say we should meet?`; the model answered with a venue and time absent from its retained context. There was no empty-history paired control here, so we do **not** attribute that answer causally to the residual `We`.

### Separate sending-counter issue on the pinned baseline

Audio reached the client and `generated_ms` advanced, but `sent_ms` stayed zero. The [Realtime forwarder](https://github.com/vllm-project/vllm-omni/blob/a3dee6f/vllm_omni/entrypoints/duplex/session_runner.py#L183) only carries the acceptance callback for `response.audio.delta`, while the [projector](https://github.com/vllm-project/vllm-omni/blob/a3dee6f/vllm_omni/entrypoints/duplex/realtime_output.py#L842) emits `response.output_audio.delta`. The callback is therefore dropped before the sender.

This is separate from the zero-audio rule: repairing the callback cannot create an audio duration when no audio was generated. The approximation uses `max(sent_ms, generated_ms)`, which explains why the two nonzero-prefix observations still match the formula. These are pinned-baseline findings, not a claim about the latest branch or attribution to a particular PR.

A historical source-only check on September 16 at 01:52 UTC found the mismatch on then-main `2185c4e`; at that time #7413 was still open at `93f53f1`. [#7413](https://github.com/vllm-project/vllm-omni/pull/7413) subsequently merged at 07:35 UTC as `99ff4f3`, deleting the old forwarding chain. These session results remain pinned to `a3dee6f`; they have not been rerun on the merged framework and do not establish replacement-path correctness. The finding is preserved for the planned Qwen integration, not another old-path repair.

<details>
<summary>Reproduction inputs and sequence</summary>

Common system message:

```text
You are a helpful assistant. Answer clearly and concisely in English.
```

First user message in each session:

```text
Invent a simple fictional day trip and describe it in exactly three short sentences, without headings or numbering. In sentence one, invent a distinctive fictional venue name and choose a meeting time there. In sentence two, choose a rainy-day backup destination. In sentence three, choose a snack. Invent the specific details yourself. Write the time in words, avoid abbreviations, and end each sentence with a period.
```

Session settings: `modalities=["text","audio"]`, `output_audio_format="pcm16"`, `turn_detection=null`, `playback_commit_policy="ack_only"`, `temperature=0`, `max_response_output_tokens=384`, and `extra_body={"native_duplex":false,"auto_response":false,"seed":1729}`. The deployment enables `duplex_session` but leaves Qwen computation in turn mode. Observed stage request settings were Thinker temperature 0 / seed 1729 / max_tokens 384, Talker 0.9 / request seed unset / 4096, and Code2Wav 0 / request seed unset / 65536. An unset request seed does not mean that engine initialization was unseeded.

For each text input, wait for `conversation.item.done`, then send `response.create`. In cancellation cases, send `playback.ack` with equal played/committed positions, immediately followed by `response.cancel`; wait for both acknowledgements before submitting the follow-up. The partial case triggers after at least six seconds of decoded mono PCM arrives. For completed generation, observe the stream returning, allow a 0.5-second receive window, then send `conversation.item.truncate` with the actual item/content indices and `audio_end_ms=4000`; wait for truncation and playback acknowledgement.

In every case, ask `Where and at what time did you say we should meet?` next and inspect its actual model input, not just whether the answer sounds plausible. Exact chunks, timing and model words may vary across runs.

</details>

## Limits and the existing E discussion

- One prompt template and one run per scenario; no policy ranking, MiniCPM result or new-framework Qwen implementation is claimed. These runs cover text input, not interrupted user-audio preservation.
- All six responses had exactly one completion event, with no further audio received for that response in the recorded sessions. A suspected early-completion issue did not reproduce here.
- The serving process returned zero on shutdown, but teardown still force-killed a Code2Wav process and reported a leaked semaphore. Successful responses do not mean clean shutdown.

The maintainer discussion relayed to us already raised three considerations: inspect how the model was trained; prefer playback-based history for voice-focused use; and consider an application-selectable policy when text is displayed faster than speech, especially for long tool-assisted answers. These are reported discussion points, not an approved configuration design. We should build on them rather than repeat the same text-versus-audio question as new.

The [Qwen3-Omni report, §§2.4 and 4](https://arxiv.org/html/2509.17765v1), describes multi-turn context and training format, but the inspected sections do not specify training examples for interrupted, partially displayed or played answers. That leaves a specific model-author question: how are those two progress states represented in training and the next turn? It is not evidence that the model was never trained on interruption.

Our proposed distinction is between tool results, client-confirmed displayed text, reported playback progress, and the history actually supplied to the next model turn. Delivered text alone is not evidence of display or reading; reported playback is not evidence of human comprehension. Stopping narration should not automatically mean deleting retrieved material or the visible conversation. Keeping a tool result also must not misrepresent it as something the assistant has already said.

For the eventual Qwen work, the concrete choices are the default for a voice-focused session, what an application must report before a display-based policy can apply, and whether the model can consume the distinction correctly. The [long-answer news comparison is now complete](news-displayed-text.md): three headings are assumed displayed while the supplied playback position lies in the first, followed by a question about the third heading or a request to continue narration. It is a separate controlled history comparison, not browser or live-cancellation evidence. A selectable production policy has not been implemented, and these experiments changed no text-only history rule, proportional approximation or production default.

AI assistance: Codex assisted with experiment code, analysis, and writing.
