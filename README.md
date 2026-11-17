# meta-coach

Audits how I use an AI coding assistant and proposes changes to my instructions file,
then measures whether those changes actually reduce friction.

## Plan

1. **Load logs into SQLite** — done (`load_logs.py`)
2. **Friction signals** — done (`signals.py` for counts, `detect_llm.py` for corrections, `label.py` to score)
3. **Daily summaries** through an LLM, so a month of history fits in context — done (`summarize.py`); a third of its friction claims misread the source, see results
4. **Evaluator** that proposes a diff to the instructions file — done (`propose.py`): the model finds leads, I accept or reject each one, and only accepted leads are merged. The model's own verifier is not trusted, see results
5. **Before/after comparison** of friction with the old and new instructions — tool done (`compare.py`), baseline recorded, waiting for enough sessions after the change. Replay is not built, see results

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
python3 compare.py             # friction before vs after the instructions file was applied
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

## Before/after results

`compare.py` splits sessions by whether they started before or after the instructions file
was created (2026-11-17 06:44 UTC), and runs the same instruments on both sides.

Baseline, the 13 sessions before the change:

| Measure | Before |
|---|---|
| Prompts I typed, scored by the detector | 159 |
| Flagged as corrections by the detector | 14 (8.8 per 100 prompts) |
| Mistakes I confirmed in review | 3 (1.9 per 100 prompts) |
| Times I interrupted the assistant | 3 |
| Failed tool calls | 47 (5.2 per 100 calls) |

After the change: 1 session and 2 prompts so far, far too few to say anything.

Read these with care:

- **This needs a lot more data than I have.** With a baseline this small, even a halving of
  the flagged rate could not be told from chance until about 420 prompts have been typed
  after the change, roughly six weeks at my current pace. Smaller effects are out of reach.
  `compare.py` prints this number and refuses to compare below 100 after-prompts.
- **It is not an experiment.** The tasks before and after differ, so a gap shows direction,
  not cause. Alternating days with and without the file would be the fair design.
- The detector is noisy (precision 0.42 on my labels). The confirmed-mistakes row is the
  trustworthy one, and it only fills in when I review the new leads (`propose.py --review`).
- Leads are dated by day, so a lead from the cutoff day counts as "before".
- **Replay is not built.** The original plan was to replay held-out tasks with the old and
  new instructions. There are no held-out tasks yet (the rules came from these same
  sessions), the incidents depend on machine state that cannot be recreated (an SSD, an
  installer mid-run), and replaying through a different model than the assistant I actually
  use would test the wrong thing.

To update: `python3 load_logs.py && python3 detect_llm.py && python3 compare.py`.

## Outside test: SWE-chat

`swechat.py` samples labelled user prompts from SWE-chat (Stanford, ~18,000 real coding-agent
sessions; gated, needs `HF_TOKEN` in `.env`) and runs the same correction detector on them.
213 prompts from 80 sessions, sampled 2026-11-17.

| SWE-chat label | Prompts | Flagged by the detector |
|---|---|---|
| correction | 80 | 17 (21%) |
| rejection | 2 | 0 |
| failure_report | 20 | 5 (25%) |
| non_pushback | 111 | 7 (6%) |

Taken at face value, recall on corrections is 0.21 and precision 0.59. Do not take it at face value:

- SWE-chat's "correction" is far looser than mine. Of 10 random "corrections" the detector
  missed, about 9 are not corrections by my definition: "Fix all", "yes do option 1",
  "What do you think?", follow-up questions. Their labels come from an LLM judge, and they
  mark 38% of prompts as corrections against about 5% in mine.
- So the low recall mostly measures the gap between the two definitions, not the detector.
- What it does show: the detector flags 6% of prompts SWE-chat calls ordinary, similar to
  its false-alarm rate on my own data, so that rate is not specific to me.
- To use SWE-chat as a real test set, a sample has to be re-labelled by my definition first.

### Re-labelled by my definition

100 of the 213 prompts, picked at random, re-labelled by Claude using my definition (the
`swechat_mine` table), not by me.

- 24 of the 100 were not typed by a person at all: skill instructions, plan templates and
  subagent reports that the agent put in the user's slot. SWE-chat labelled 5 of them
  "correction". The detector flagged 3.
- Of the 76 typed prompts, 4 are corrections and 4 borderline (my change did not work,
  questioning a claim of mine, disliking a design). About 5 to 10%, close to my own rate.
- SWE-chat's own labels against these: every real correction is marked "correction", but so
  are 31 that are not (precision 0.11).

| Detector against the re-labelled prompts | Precision | Recall |
|---|---|---|
| borderline counted as "not" | 0.29 | 0.50 (2 of 4) |
| borderline counted as correction | 0.57 | 0.50 (4 of 8) |

Close to its numbers on my own data (0.42 / 0.71), with only 4 to 8 positives, so the
range is wide. The useful finding is the size of the problem: corrections are rare in other
people's sessions too, and any count of them needs injected text filtered out first.
