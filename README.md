# meta-coach

Audits how I use an AI coding assistant and proposes changes to my instructions file,
then measures whether those changes actually reduce friction.

## Plan

1. **Load logs into SQLite** — done (`load_logs.py`)
2. **Friction signals** — done (`signals.py` for counts, `detect_llm.py` for corrections, `label.py` to score)
3. **Daily summaries** through an LLM, so a month of history fits in context — done (`summarize.py`); a third of its friction claims misread the source, see results
4. **Evaluator** that proposes a diff to the instructions file — done (`propose.py`): the model finds leads, I accept or reject each one, and only accepted leads are merged. The model's own verifier is not trusted, see results
5. Replay held-out tasks with old vs new instructions and compare the friction numbers

## Run

```bash
python3 load_logs.py --check   # self-test
python3 load_logs.py           # reads ~/.claude/projects, writes coach.db
python3 signals.py             # per-session friction counts
python3 label.py --score       # keyword detector vs hand labels
python3 detect_llm.py          # LLM detector vs hand labels (needs NVIDIA_API_KEY in .env)
python3 summarize.py --dry-run # what the summariser would send, without sending it
python3 summarize.py           # one summary per day, cached; prints the digest
python3 propose.py --dry-run   # the leads and the source lines each would be checked against
python3 propose.py --review    # I accept or reject each lead (a / r / q); resumes where it stopped
python3 propose.py             # verify leads, merge the accepted ones, write proposal.diff (never edits the real file)
```

Every script has a `--check` self-test. No dependencies beyond Python 3.9.

**What leaves the machine.** `detect_llm.py`, `summarize.py` and `propose.py` send text to NVIDIA's API.
Before sending, key-shaped strings, emails and phone numbers are masked, pasted files are
replaced by `[PASTED TEXT]`, and app-injected blocks are dropped. Tool calls and tool output
are never sent. Names of people are not masked. `coach.db` holds private chat history and
is git-ignored.

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

## Daily summary results

9 days, 125,553 characters of prompts and replies → 4,378 characters of summary (29x smaller).
Model: `mistralai/mistral-large-4-1t-a52b` on NVIDIA NIM, reasoning effort low.

| Check | How | Result |
|---|---|---|
| Quotes that appear word for word in what I typed | automatic | 20 of 20 |
| Hand-labelled corrections that the summaries quote | automatic | 4 of 7 |
| Friction claims that hold up when the source is read in context | read by Claude, not by me | 6 of 9 |

Read these with care:

- A real quote does not make the claim true. All 3 wrong friction claims carry a real quote
  but misread who was at fault. Example: I said "done" too early, the assistant correctly
  reported the install was not finished, and the summary says the assistant misread the terminal.
- So step 4 must treat each bullet as a lead and re-read the turn behind its quote before
  proposing anything. The quote is the pointer back to that turn.
- A first version of the prompt got 7 of 12 friction claims right (one more partly right) and
  quoted 5 of 7 corrections. Tightening the prompt removed some errors and added others.
  With this few items the two versions cannot be told apart, and the second was written
  after seeing the first one's errors on these same days.
- Two fixes that were about missing input, not wording: my own terminal commands are now shown
  to the model (it had blamed the assistant for one), and briefs written by another session
  are dropped (the model had quoted one as mine).
- Preference items were read but not checked against the source.
- Hidden reasoning counts against `max_tokens`. At default effort the model used 4,096 tokens
  on a 600-character day and returned nothing. Low effort with 8,192 tokens works, at one to
  two minutes per day.
- Not repeatable run to run, even at temperature 0: the same day came back empty once and
  complete on the retry.

## Evaluator results

Leads come from two places: the friction and preference bullets of the daily summaries, and
the messages the correction detector flagged. 30 in all, 26 after dropping flagged messages
that a friction bullet already covers.

**The model's verifier.** Each lead was checked by the model against the six lines of
conversation around its quote.

| Check | Result |
|---|---|
| Verifier agrees with my hand labels, 12 detector-flagged messages | 3 of 12 |
| – real corrections it kept | 2 of 5 |
| – false alarms it rejected | 1 of 7 |
| Verifier agrees with my review of all 26 leads (below) | 8 of 26 |
| – leads I accepted that it kept | 4 of 6 |
| – leads I rejected that it rejected | 4 of 20 |

It said "supported" to 20 of 26 leads and I accepted 6. It is not a usable filter, so its
verdicts are ignored once a human review exists.

**My review** (2026-11-17, all 26 leads; 7 carried over from the nine-message review above).

| Source of lead | Leads | I accepted |
|---|---|---|
| Friction bullets from the summaries | 9 | 2 |
| Preference bullets from the summaries | 9 | 3 |
| Detector-flagged messages not covered by a bullet | 8 | 1 |
| Total | 26 | 6 |

**The proposal.** The 6 accepted leads were merged into 5 lines (`proposal.diff`, against
`~/.claude/CLAUDE.md`, which did not exist yet). Nothing is applied automatically.

**Applied** (2026-11-17, by my decision): 4 of the 5 lines, creating `~/.claude/CLAUDE.md`.
I skipped the line telling the assistant to change all other items "without asking for
confirmation": too broad for one incident. Sessions before this date ran with no global
instructions file, which is the baseline for step 5.

Read these with care:

- 6 real leads out of 26 means the automatic part of the pipeline is a lead generator, not
  a judge. The human pass takes a few minutes and is what makes the proposal trustworthy.
- Claude's own reading of the 9 friction claims agreed with mine on only 5. Deciding who
  was at fault from a transcript is hard for a careful reader too; I am the ground truth.
- Each proposed line rests on one or two incidents. Whether the lines actually reduce
  friction is what step 5 has to measure.
- The merge step first dropped 3 of my 6 accepted leads as "too narrow". It is now told to
  keep or reword every confirmed lead, and it prints any it leaves out. The cost: reworded
  lines can come out more general than the incident justifies, so read the diff before applying.
- The verifier is asked "was the assistant at fault?", my labels answer "did I push back?".
  Any future judge should be tested against the 26 reviewed leads, on new sessions.
