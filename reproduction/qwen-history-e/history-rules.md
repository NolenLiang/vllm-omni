# Qwen3-Omni interrupted-answer history — two real-model comparisons

Prepared evidence for the existing [Item E discussion](https://github.com/vllm-project/vllm-omni/issues/7055), not a new assignment or a production change. See the [packet overview](README.md) for scope and publication limits.

## Motivation and proposed direction

Following the [Item E discussion](https://github.com/vllm-project/vllm-omni/issues/7055#issuecomment-5677251141), we compared retaining an interrupted answer's prefix with deleting the answer. Our provisional recommendation is narrow: preserve information already provided to the user when constructing the next model input; interruption alone should not erase it. Keep the existing approximate playback-to-text truncation for now. These runs do not establish that an additional interruption note should become the default.

## Method

Qwen3-Omni-30B-A3B-Instruct `26291f7`, vLLM-Omni `a3dee6f`, official vLLM 0.29.0 ARM64, two GB200 GPUs. Eager execution, prefix caching disabled; Thinker on one GPU, Talker and Code2Wav on the other.

Two distinct generated day-trip answers supplied the reference material. For each, we reused the same first answer across three conditions:

1. Retain exactly its first sentence as an assistant message.
2. Remove that assistant answer entirely, retaining the original user request.
3. Retain the same sentence and append two newlines followed by `[Playback note: The answer was interrupted here; the remaining content was not played.]` in that assistant message.

The common system message was `You are a helpful assistant. Answer clearly and concisely in English.` Each condition answered both `Where and at what time did you say we should meet?` and `I did not hear the rainy-day part. Where can we go if it rains?`, three times in shuffled order: 18 continuations per dialogue, 36 total. These are repeatability runs of two dialogues, not 36 independent samples.

The first fixture used request temperature 0 / seed 1729; the second used 0.8 / 2718 and a different first prompt. Every continuation used temperature 0 / seed 1729, max_tokens 384, requesting text and WAV audio. These request controls do not override the Talker's deployment defaults. The two fixtures are separate scenarios, not an isolated sampling comparison.

All reference and continuation requests completed with text and audio. Returned input text and tokens were checked separately from client requests; this is returned-input evidence, not an internal session-history trace. Within each condition, all three texts matched while the three audio waveforms differed. Qualitative assessment was label-blinded AI text review, followed by inspection of the actual input mapping; no human review or audio quality assessment is claimed.

<details>
<summary>Exact text-history inputs for reproducing the comparison</summary>

The first user message for dialogue 1 was:

```text
Invent a simple fictional day trip and describe it in exactly three short sentences, without headings or numbering. In sentence one, choose a meeting place and time. In sentence two, choose a rainy-day backup destination. In sentence three, choose a snack. Invent the specific details yourself. Write the time in words, avoid abbreviations, and end each sentence with a period.
```

Dialogue 2 replaced only `choose a meeting place and time` in that prompt with `invent a distinctive fictional venue name and choose a meeting time there`.

Use these fixed reference prefixes rather than regenerating each condition's reference:

```text
Dialogue 1: We will meet at the old stone bridge by the river at quarter past ten in the morning.
Dialogue 2: We'll meet at the Moonbeam Diner at half past eleven.
```

Construct each `/v1/chat/completions` request as the common system message, that dialogue's original user message, the condition's assistant message (absent for deletion), and one of the two follow-up user questions above. Request `modalities: ["text", "audio"]`, `audio: {"format": "wav"}`, `stream: false`, `n: 1`, `temperature: 0`, `seed: 1729`, `max_tokens: 384`, `return_prompt_text: true`, and `return_token_ids: true`. Preserve raw responses and inspect the returned prompt, not only the answer. Reproducing these messages does not guarantee identical GPU output across environments.

</details>

## Observations

| History supplied | Dialogue 1: bridge, quarter past ten | Dialogue 2: Moonbeam Diner, half past eleven |
| --- | --- | --- |
| Prefix | Meeting correct 3/3; no meeting change in rain answer 3/3 | Meeting correct 3/3; no meeting change in rain answer 3/3 |
| Entire answer removed | Original place/time regenerated, with extra content, 3/3; conflicting noon meeting in rain answer 3/3 | Both place and time changed in meeting answer 3/3; conflicting place and noon meeting in rain answer 3/3 |
| Prefix + neutral note | Meeting correct 3/3; no meeting change in rain answer 3/3 | Meeting correct 3/3; no meeting change in rain answer 3/3 |

Example from dialogue 2: the reference was `We'll meet at the Moonbeam Diner at half past eleven.` After deletion, the direct recall answer began `We should meet at the Whispering Pines Treehouse Café at noon.` Retaining the sentence, with or without the note, preserved the reference place and time.

A different rainy-day suggestion is not automatically wrong: the user asked for unheard information. The problem is silently changing an already supplied meeting arrangement. Conversely, dialogue 1's correct answer after deletion does not prove preserved memory: the checked input contained no reference place/time. The model may have regenerated common details; the cause is not established.

## Limits and open decision

- These are stateless Chat history comparisons, **not live Realtime cancellation, physical playback, or a new-framework Qwen integration**. The selected first sentences are controlled text prefixes. Only dialogue 1's provisional audio clip received an auxiliary ASR check: ASR transcribed that clip as the first sentence, but this is not human-verified alignment. Dialogue 2's audio boundary is unverified.
- Two short, related fictional scenarios cannot establish a general error rate or a best rule. The original user request remains after assistant deletion and may cause the model to regenerate a trip. Neither scenario establishes a benefit from adding the neutral note.
- An earlier startup attempt stopped before model loading because of a GPU-visibility assertion; it supplied no model result. Both completed runs still reported forced Stage 2 termination and a leaked-semaphore warning during teardown; successful responses do not imply clean shutdown.
- This work did not change the production approximation. [#7413](https://github.com/vllm-project/vllm-omni/pull/7413) merged as `99ff4f3` on September 16 and removed the old Qwen fallback path. Applying these observations requires agreement on how Qwen sessions construct the next model input; the implementation path has not been selected here, and this is not another change to the removed path.

For E, does preserving the played-prefix approximation remain the preferred starting rule, without adding a neutral interruption note by default until it demonstrates a concrete benefit? The [completed live-session follow-up](realtime-sessions.md) records actual next-turn inputs after cancellation or truncation; the [completed news comparison](news-displayed-text.md) separately examines displayed text and retained tool results. Neither establishes replacement-path correctness.

AI assistance: Codex assisted with experiment code, analysis, and writing.
