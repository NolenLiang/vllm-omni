# Interrupted Qwen history — tool results, displayed text, and playback

Completed controlled-model comparison and discussion material for Item E in [#7055](https://github.com/vllm-project/vllm-omni/issues/7055), not a separate upstream RFC. This is not a claim to the whole Qwen integration or a proposal to change the production default immediately. See the [packet overview](README.md) and the separate [live-session evidence](realtime-sessions.md).

## Problem and proposed scope

A long answer can be fully displayed while speech has only reached its first heading. After interruption, the user may ask about the third displayed heading or ask the assistant to continue reading. Treating all history as either “already heard” or “must be deleted” cannot express that situation accurately.

Three maintainer considerations were relayed to us and remain separate: investigate training, consider a voice-first default, and consider an application-selectable rule. None implies agreement on the other two.

Our proposed scope for the eventual Qwen session work is:

1. Keep completed tool results distinct from the assistant answer being shortened. Stopping speech should not itself delete the retrieved material or imply that a completed tool action was undone. Existing explicit deletion and retention limits still apply.
2. Distinguish generated text, application-confirmed displayed text and reported playback position. Receipt is not display; display is not reading; a playback report does not establish human comprehension. Missing display information remains unknown, not “the whole answer was shown.”
3. Construct the next model input from the chosen rule without presenting retained tool material as something the assistant already said. Keep the current proportional playback-to-text approximation; choosing a default and exposing a public option are separate decisions.
4. Validate that the model actually receives the selected history. Changing a framework list is insufficient if the model retains different context internally. Do not advertise a choice a model integration cannot implement.

This does not add a configuration field, dictate a universal default or implement browser reporting. [#7413](https://github.com/vllm-project/vllm-omni/pull/7413) has merged as `99ff4f3` and removed the old Qwen fallback; these fixed-baseline observations are preparation for its replacement, not an extension of the removed path. History retention is not exact resumption of speech.

## Completed real-model observation

Environment: Qwen3-Omni-30B-A3B-Instruct `26291f7`, Omni `a3dee6f`, official vLLM 0.29.0 ARM64 image with additional fixed runtime dependencies, including Transformers 5.14.1 and tokenizers 0.22.2; two GB200 GPUs. Eager execution, prefix caching disabled; Thinker on one GPU and Talker/Code2Wav on the other. Startup uses the [pinned Omni import patches](https://github.com/vllm-project/vllm-omni/blob/a3dee6f/vllm_omni/__init__.py#L21); this is not a claim of an otherwise unmodified image, and their effect was not isolated.

One model-generated briefing and its WAV defined the common reference; continuation requests supplied text history, not that WAV. The reference contained 1,121 characters and 89.016875 seconds of mono 24 kHz PCM16. The predeclared five-second cursor produced the existing proportional prefix:

```text
1. Night ferry begins a six-week trial, published by Harbor Le
```

The partial word was not repaired. This is a time-based estimate, not verified word/audio alignment.

Each continuation kept the same pre-injected tool call and synthetic tool result, and received the same application-state explanation: the whole answer was displayed, but audio stopped at five seconds, with the prefix above as the approximate boundary. Only the preceding assistant answer changed: prefix versus full displayed text. Two questions, three stability repetitions each, yielded 12 continuations of **one** fixture, not 12 independent dialogues.

| Retained assistant answer | “Source of the third headline?” | “Continue reading from where audio stopped.” |
| --- | --- | --- |
| Approximate played prefix | Correct publisher and report code 3/3 | Restarted at the first heading and rephrased the briefing from retained records 3/3 |
| Full displayed text | Same correct answer 3/3 | Restarted at the first heading and reproduced the displayed briefing verbatim 3/3 |

The displayed order happened to match the record order, so the source could be reconstructed without remembering the generated display. The cursor was near the start; returning to the first sentence repeats a short prefix, not necessarily an unacceptable amount of speech. Neither choice demonstrated exact resumption, and these observations do **not** establish a winning default or prove a new option is necessary.

The original displayed briefing itself omitted the records' per-household qualification for the seed allowance. Verbatim reproduction retained that wording, while the prefix condition restated the household restriction. Fidelity to displayed wording and fidelity to source facts are different questions; this is not a general factual-accuracy comparison.

All 13 requests, including the common briefing, completed with text and audio. The returned processed prompts preserved the supplied call and tool result; their token IDs were independently decoded and matched. This is returned-input verification, not an internal engine tracing claim. Assessment was policy-label-blinded AI text review, not human listening. No real search, tool execution, browser rendering, physical playback or live cancellation occurred in this comparison.

Shutdown returned zero but still force-killed the remaining Code2Wav process and reported one leaked semaphore. Successful responses do not establish clean teardown.

<details>
<summary>Fixed inputs and reproduction conditions</summary>

Use a system message: `You are a helpful assistant. Answer clearly in English using the supplied fictional news records. Do not invent facts or sources.`

The first user request is:

```text
Using the three fixed fictional records from the simulated news lookup, prepare one news briefing for both display and spoken delivery. Include every record once, in any presentation order you choose. Use exactly three numbered sections, beginning on separate lines with '1. ', '2. ', and '3. '. Begin each section with that record's exact title, then give its details and identify its publisher and report code. Use about one hundred eighty to two hundred thirty words in total. Do not add an introduction, conclusion, new information, or a request to search again. Use plain text, without bold markers or Markdown heading characters.
```

Append an assistant message with empty content and one supplied `search_news` tool call, arguments `{"query":"fictional Dunmere news"}`, ID `call_news_1`; then a tool message with that ID and the following JSON content. These messages are a fixture, not an executed call. No new `tools` or `tool_choice` is sent.

```json
{
  "fictional": true,
  "records": [
    {
      "id": "A",
      "title": "Night ferry begins a six-week trial",
      "publisher": "Harbor Ledger",
      "report_code": "FERRY-482",
      "facts": "The fictional town of Dunmere will test a night ferry for six weeks. It runs on Fridays and Saturdays between Lantern Pier and Reed Bank. The final departure is at eleven in the evening. A one-way ticket costs four credits. The council will publish passenger counts after the trial; no permanent service has been approved."
    },
    {
      "id": "B",
      "title": "Seed library adds weekend collection",
      "publisher": "Juniper Bulletin",
      "report_code": "SEED-731",
      "facts": "Dunmere's seed library will open on Saturday mornings for eight weeks. Visitors can collect bean, pea and sunflower seeds free of charge. Each household may take two packets per visit. Volunteers will offer planting guidance. Donations are welcome but are not required, and there is no reservation system."
    },
    {
      "id": "C",
      "title": "Footbridge closes for timber repairs",
      "publisher": "Copper Finch Review",
      "report_code": "BRIDGE-956",
      "facts": "The Alder Footbridge will close for twelve days to replace damaged timber. Pedestrians should use the marked path beside the old mill, which adds seven minutes to the walk. Bicycles must be walked along that path. The repair does not affect the nearby road bridge. The reopening date depends on a final safety inspection."
    }
  ]
}
```

Make one `/v1/chat/completions` request with `modalities=["text","audio"]`, WAV output, `stream=false`, `n=1`, `temperature=0`, `seed=1729`, `max_tokens=768`, `return_prompt_text=true`, `return_token_ids=true`. Save its complete text/audio and verify the tool history survived preprocessing. Do not independently regenerate the reference answer for each policy.

Compute `total_ms = frames * 1000 // sample_rate`, then `prefix = text[:int(len(text) * min(1.0, 5000 / max(1, total_ms)))].rstrip()`. Save the first five seconds of original PCM without claiming alignment. Require an interior cutoff in the first section; preserve any failure rather than retrying for a preferred reference.

For all continuations append the following to the common system message, replacing `<JSON-encoded prefix>` with the same prefix string:

```text


Application state for this controlled scenario: the entire previous briefing was displayed on the screen. Audio playback was interrupted at five seconds; the displayed briefing has NOT all been spoken. The existing time-based approximation maps the played audio to this text prefix (not an exact word alignment):
<JSON-encoded prefix>
The original supplied records remain available.
```

Keep the original user request and tool-history messages, add either the prefix or full reference as the assistant answer, then ask one of:

- `What is the source of the third headline on the screen? Give the publisher and report code.`
- `Please continue reading from where the audio stopped.`

Use the same request settings for every continuation. These fields control the Thinker; Talker uses deployment defaults, not a guaranteed shared seed. Run each condition three times in shuffled order and preserve every output. The state explanation is experiment input, not a proposed mechanism for trusting arbitrary client text as system instructions. Exact generated words and audio may differ across environments.

</details>

## Decisions requested, separately

- **Training:** is there a public example of how partially displayed/played interruptions are represented in the next model turn? The inspected [Qwen3-Omni report](https://arxiv.org/html/2509.17765v1) does not specify that rule; this does not establish absence of relevant training.
- **Default:** what behavior should a voice-focused Qwen session promise? Our proposal keeps the current approximation while this is decided, rather than interpreting one experiment as approval to change it.
- **Optional behavior:** is preserving application-confirmed displayed text a supported use case for the planned Qwen integration? If yes, its reporting requirements and model support need to be defined before promising a public option.

AI assistance: Codex assisted with experiment code, analysis, and drafting.
