# meta-coach

Audits how I use an AI coding assistant and proposes changes to my instructions file,
then measures whether those changes actually reduce friction.

## Plan

1. **Load logs into SQLite** — done (`load_logs.py`)
2. **Friction signals** — done (`signals.py` for counts, `detect_llm.py` for corrections, `label.py` to score)
3. Daily summaries through an LLM, so a month of history fits in context
4. Evaluator agent that proposes a diff to the instructions file
5. Replay held-out tasks with old vs new instructions and compare the friction numbers

## Run

```bash
python3 load_logs.py --check   # self-test
python3 load_logs.py           # reads ~/.claude/projects, writes coach.db
python3 signals.py             # per-session friction counts
python3 label.py --score       # keyword detector vs hand labels
python3 detect_llm.py          # LLM detector vs hand labels (needs NVIDIA_API_KEY in .env)
```

Every script has a `--check` self-test. No dependencies beyond Python 3.9.
`detect_llm.py` sends prompts to NVIDIA's API after masking keys, emails and phone numbers. `coach.db` holds private chat history and is git-ignored.

## Data

One row per message block in `messages`:

| column | meaning |
|---|---|
| `kind` | `prompt` (me), `reply`, `tool_use`, `tool_result` |
| `tool`, `is_error` | tool name; whether the tool call failed |
| `session_id`, `ts`, `cwd` | which session, when, which project folder |

## Known limits

- Some `prompt` rows are text the app injected (reminders, pasted files), not typed by me.
  Step 2 has to filter these before counting corrections.
- Subagent messages and the model's hidden reasoning are not loaded.

## Labels

`label.py` stores hand labels (correction / not) in the `labels` table and scores the detector.
The first pass (142 prompts, 7 corrections) was labelled by Claude reading each prompt with the
preceding reply, not by me. Borderline calls went to "not". Re-label with `python3 label.py`
after `DELETE FROM labels` to replace them with my own.

On 2026-11-16 I reviewed the 9 prompts where the LLM detector and these labels disagreed and
kept all 9 labels. The other 133 labels are still Claude's and unreviewed.

## Correction detection results

142 labelled prompts, 7 of them corrections.

| Detector | Precision | Recall | Hits | False alarms | Missed |
|---|---|---|---|---|---|
| Keyword rules (`signals.py`) | 1.00 | 0.14 | 1 | 0 | 6 |
| LLM, zero-shot (`detect_llm.py`, `mistralai/mistral-large-4-1t-a52b` on NVIDIA NIM) | 0.42 | 0.71 | 5 | 7 | 2 |

Read these with care:

- 7 positives is tiny. One label changing moves recall by 14 points.
- I reviewed the 9 disagreements and all 9 labels stood, so the 7 false alarms and 2 misses
  are the model's errors, not label noise. The 133 prompts where the model and Claude agreed
  are unreviewed.
- The false alarms: two "this step is stuck" reports, an answer to the assistant's question,
  a declined suggestion, a new request, an accepted plan, and me changing my own answer.
  The misses are both a short fact correction inside an otherwise ordinary message
  ("macfuse install i have already put the password").
- One "the same step failed again" message is labelled a correction while the two "this step
  is stuck" reports are not. The line between them is thin.
- One model, one run, temperature 0. No second model compared yet.
- The model reasons before answering, about 10 seconds per prompt. With `max_tokens` at 512
  it ran out before answering on 18 prompts; 4096 fixed that.
