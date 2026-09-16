# Qwen interrupted-answer history: Item E evidence

Prepared September 16, 2026. This packet supports the existing [Item E discussion in #7055](https://github.com/vllm-project/vllm-omni/issues/7055), not a new upstream RFC or an assignment claim.

## Read the three evidence types separately

| Material | What actually ran | What it establishes |
| --- | --- | --- |
| [Two text-history comparisons](history-rules.md) | Two generated day-trip references; three history conditions, two questions and three repeats each: 36 continuations | How supplied text history changes these continuations; returned model-input text and tokens were checked |
| [Live Realtime sessions](realtime-sessions.md) | Three sessions, one run per cancellation/truncation scenario; client-supplied playback positions | The actual next request and processed engine prompt on the old fixed path; the zero-audio playback-only expectation failed |
| [Displayed news and retained tool results](news-displayed-text.md) | One generated briefing and 12 continuations using pre-injected fictional tool records and application state | Prefix versus full displayed-text observations under the same supplied records and state; not real searching or tool execution |

The tables' `3/3` counts mean three stability repeats of one condition, not three independent dialogues or a general success rate. The Realtime scenarios were each run once and must not be relabeled as three repeated passes.

## Shared environment and present-day scope

The main experiments used Qwen3-Omni-30B-A3B-Instruct `26291f7`, Omni `a3dee6f`, official vLLM 0.29.0 ARM64 and two GB200 GPUs, with eager execution and prefix caching disabled. Each document preserves its additional settings, observations and limitations. Request controls did not make the generated audio deterministic.

[#7413](https://github.com/vllm-project/vllm-omni/pull/7413) merged on September 16 as `99ff4f3` and removed the old Qwen fallback path. The reported runs have not been repeated on its replacement. Old source links are intentional fixed-baseline evidence, not claims that those call sites still exist on main.

## What this packet includes and excludes

- It contains reproduction inputs, conditions, aggregate counts, representative output excerpts, failed expectations and shutdown limitations. It is not the complete raw-log archive, an executable deployment bundle, or a promise of byte-identical reruns.
- Saved generated audio informed duration calculations; no player-output recordings or audio files are included here. No physical playback, browser display or human listening was measured. The auxiliary speech-recognition check covers only the first day-trip clip and is not human verification.
- Qualitative assessments were AI text reviews with history-condition labels hidden before assessment, followed by separate input/mapping checks. No completed human review, listening assessment, broad quality score or policy winner is claimed.
- The news comparison is complete. A public history-policy option and the new-framework Qwen integration are not implemented by this packet. The existing production approximation was not changed.
- Cite this packet through a fixed commit on our own dedicated evidence branch, separate from the production output-delivery PR. Updating the existing E discussion or notifying contributors requires the user's approval.

The training question, a voice-focused default, and possible application-selected behavior remain separate decisions. The evidence does not settle any of them by itself.

AI assistance: Codex assisted with experiment code, evidence analysis, and writing. Qualitative review was AI-only; no human review is claimed.
