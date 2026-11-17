"""Step 4: check each lead against its source, then propose lines for the instructions file.

A lead is a friction or preference bullet from the daily summaries (summarize.py), or a
message the correction detector flagged (detect_llm.py). Step 3 showed that about a third
of the summary claims misread the conversation, so nothing is taken on trust:

  1. verify  one LLM call per lead, given only the turn its quote points to: is the claim
             supported, and what general rule would have prevented it?
  2. review  I accept or reject each lead myself. The model's verifier agreed with my hand
             labels on 3 of 12 messages, so once I have reviewed, only my verdicts count.
  3. merge   one LLM call that folds the accepted leads into at most 8 lines. Each line must
             cite the leads it came from; a line that cites none is dropped.

Usage:  python3 propose.py                 # verify, merge, write proposal.diff
        python3 propose.py --review        # show me each lead: a = accept, r = reject, q = quit
        python3 propose.py --dry-run       # list the leads and their source, send nothing
        python3 propose.py --file <path>   # instructions file to diff against
        python3 propose.py --check         # self-test, no network
The instructions file is only read. The proposal goes to proposal.diff for me to review.
"""
import contextlib
import difflib
import hashlib
import io
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

from detect_llm import DEFAULT_MODEL, PASTE, api_key, ask, scrub
from signals import INJECTED
from summarize import norm, transcripts

TARGET = "~/.claude/CLAUDE.md"
HEADING = "## Learned from my usage (proposed by meta-coach)"
FLAGGED = "A detector flagged the marked message as the user correcting the assistant."

VERIFY = """You check one claim about a conversation between a user and an AI coding assistant.
You get the claim and the lines of the conversation it is about. The line marked >> is the
user's message the claim points to. Decide from these lines alone.

- KIND FRICTION or CORRECTION: was the assistant at fault? At fault means a wrong assumption,
  ignoring something the user had already said, doing more or less than asked, or repeating
  a step that had failed. It is not at fault when the user answers its question, asks for
  something new, made the slip themself, or when a tool, website or install failed by itself.
- KIND PREFERENCE: did the user state or clearly show something that should also hold in
  later sessions? A requirement of that day's task alone does not count.

Reply in exactly two lines:
VERDICT: SUPPORTED or NOT SUPPORTED
RULE: one imperative sentence of at most 25 words for the assistant's standing instructions,
which would have prevented this or honours the preference. It must make sense in other
sessions and name no file, person or project from this one. Write NONE if the verdict is
NOT SUPPORTED or no general rule follows."""

MERGE = """You maintain the standing instructions file of an AI coding assistant.
You get the current file and numbered candidates, each from one incident the user confirmed:
either a rule, or a description of the incident for you to turn into a rule.
Write the lines to add to the file.

- The user confirmed every candidate, so each one must end up in some line. Merge candidates
  that say the same thing, and reword a narrow one so it also applies outside its task.
  Drop a candidate only if the file already covers it.
- At most 8 lines, the best supported first. Each line is "- " plus one imperative sentence.
- End every line with the numbers of the candidates it came from, in brackets: [2, 5].
Output only those lines."""


def bullets(summary):
    """[(kind, claim, quote)] for the FRICTION and PREFERENCES bullets of one summary."""
    out, kind = [], None
    for line in summary.replace("“", '"').replace("”", '"').splitlines():
        line = line.strip()
        if re.match(r"(WORK|FRICTION|PREFERENCES):", line):
            kind = {"FRICTION": "FRICTION", "PREFERENCES": "PREFERENCE"}.get(line.split(":")[0])
        elif kind and line.startswith("- ") and not line.lower().startswith("- none"):
            quotes = re.findall(r'"([^"]{4,})"', line)
            out.append((kind, line[2:], quotes[-1] if quotes else None))
    return out


def leads(db, model):
    """[(day, kind, claim, quote, truth)]. truth is the hand label where one exists, else None."""
    out = [(day, kind, claim, quote, None)
           for day, summary in db.execute("SELECT day, summary FROM summaries WHERE model=? "
                                          "ORDER BY day, part", (model,))
           for kind, claim, quote in bullets(summary)]
    for day, text, truth in db.execute(
            "SELECT date(m.ts,'localtime'), m.text, l.is_correction FROM llm_preds p "
            "JOIN messages m USING (uuid, idx) LEFT JOIN labels l USING (uuid, idx) "
            "WHERE p.model=? AND p.is_correction ORDER BY m.ts", (model,)):
        typed = norm(scrub(INJECTED.sub("", PASTE.sub("[PASTED TEXT]", text))))
        out.append((day, "CORRECTION", FLAGGED, " ".join(typed.split()[-12:]), truth))
    return out


def context(lines, quote, before=4, after=1):
    """The transcript lines around the USER line holding `quote`, that line marked >>."""
    for i, line in enumerate(lines):
        if quote and line.startswith("USER: ") and norm(quote) in norm(line):
            return "\n".join((">> " if j == i else "   ") + lines[j]
                             for j in range(max(0, i - before), min(len(lines), i + after + 1)))
    return None


def parse(answer):
    """(supported, rule): True/False/None, and the rule text or None."""
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
    verdict = re.search(r"VERDICT:\s*(NOT\s+)?SUPPORTED", answer, re.I)
    rule = re.search(r"RULE:\s*(.+)", answer, re.I)
    rule = rule.group(1).strip() if rule else ""
    ok = None if not verdict else not verdict.group(1)
    return ok, rule if ok and rule and not rule.upper().startswith("NONE") else None


def merged(answer, n):
    """[(line, [candidate numbers])] from the merge reply; lines citing no valid candidate are dropped."""
    out = []
    for line in re.sub(r"<think>.*?</think>", "", answer, flags=re.S).splitlines():
        found = re.match(r"\s*-\s+(.*?)\s*\[([\d,\s]+)\]\s*$", line)
        refs = [int(x) for x in found.group(2).split(",") if x.strip()] if found else []
        refs = [r for r in refs if 1 <= r <= n]
        if refs:
            out.append(("- " + found.group(1), refs))
    return out[:8]


def diff(old, lines, name):
    new = old.rstrip("\n") + ("\n\n" if old.strip() else "") + HEADING + "\n\n" + "\n".join(lines) + "\n"
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        name, name + " (proposed)"))


def key_of(model, system, turn):
    return hashlib.sha1("\0".join((model, system, turn)).encode()).hexdigest()


def stored(db, model, system, turn):
    row = db.execute("SELECT answer FROM checks WHERE key=?", (key_of(model, system, turn),)).fetchone()
    return row[0] if row else ""


def cached(db, model, system, turn, key, **extra):
    """ask(), remembered in coach.db by a hash of the whole request."""
    answer = stored(db, model, system, turn)
    if not answer:
        answer = ask(model, key, turn, system, reasoning_effort="low", **extra)
        if answer.strip():  # an empty answer is not stored, so the next run asks again
            db.execute("INSERT OR REPLACE INTO checks VALUES (?,?)", (key_of(model, system, turn), answer))
            db.commit()
    return answer


def marked(lead):
    """The user's message a lead points to: the >> line of its source lines."""
    return next(line for line in lead[5].splitlines() if line.startswith(">> "))


def rkey(lead):
    return hashlib.sha1("\0".join((lead[0], lead[1], marked(lead))).encode()).hexdigest()


def unique(found):
    """Drop a flagged message that a friction bullet already covers: same message, vaguer claim."""
    covered = {(t[0], marked(t)) for t in found if t[1] == "FRICTION"}
    return [t for t in found if t[1] != "CORRECTION" or (t[0], marked(t)) not in covered]


def gather(db, model):
    """Leads that have a source turn: (day, kind, claim, quote, truth, text sent to the verifier)."""
    days, out = transcripts(db), []
    for day, kind, claim, quote, truth in leads(db, model):
        lines = context(days.get(day, []), quote)
        if lines is None:
            print(f"  no source found, skipped ({day} {kind}): {claim[:70]}")
        else:
            out.append((day, kind, claim, quote, truth, f"KIND: {kind}\nCLAIM: {claim}\nLINES:\n{lines}"))
    return out


def review(db, model, todo):
    """Ask me about each lead not reviewed yet. Stops at q or end of input; resumes next time."""
    for n, lead in enumerate(todo, 1):
        if db.execute("SELECT 1 FROM reviews WHERE key=?", (rkey(lead),)).fetchone():
            continue
        rule = parse(stored(db, model, VERIFY, lead[5]))[1]
        print(f"\n[{n} of {len(todo)}] {lead[0]}\n{lead[5]}" + (f"\nSUGGESTED RULE: {rule}" if rule else ""))
        try:
            answer = input("Is this real? a = accept, r = reject, q = quit: ").strip().lower()
        except EOFError:
            break
        if answer not in ("a", "r"):
            break
        db.execute("INSERT INTO reviews VALUES (?,?)", (rkey(lead), answer == "a"))
        db.commit()
    mine = [ok for lead in todo for (ok,) in db.execute("SELECT ok FROM reviews WHERE key=?", (rkey(lead),))]
    print(f"\nreviewed {len(mine)} of {len(todo)} leads, accepted {sum(map(bool, mine))}")


def run(db, model, path, dry=False, ask_me=False):
    db.execute("CREATE TABLE IF NOT EXISTS checks (key PRIMARY KEY, answer)")
    db.execute("CREATE TABLE IF NOT EXISTS reviews (key PRIMARY KEY, ok)")
    todo = gather(db, model)
    if ask_me:
        return review(db, model, unique(todo))
    print(f"{len(todo)} leads with a source turn, {sum(len(t[5]) for t in todo)} chars to send")
    if dry:
        for t in todo:
            print(f"\n--- {t[0]}\n{t[5]}")
        return
    key = api_key()
    # ponytail: 4 requests at a time, sqlite written from this thread only. Each call is slow
    # (hidden reasoning), not the rate limit; raise the worker count if the API allows.
    fresh = [t[5] for t in todo if not stored(db, model, VERIFY, t[5])]
    with ThreadPoolExecutor(4) as pool:
        for turn, answer in zip(fresh, pool.map(lambda turn: ask(model, key, turn, VERIFY,
                                                                 reasoning_effort="low"), fresh)):
            if answer.strip():
                db.execute("INSERT OR REPLACE INTO checks VALUES (?,?)", (key_of(model, VERIFY, turn), answer))
                db.commit()
            print("." if answer.strip() else "?", end="", flush=True)
    results = [t + parse(stored(db, model, VERIFY, t[5])) for t in todo]
    for day, kind, claim, quote, truth, _, ok, rule in results:
        mark = {True: "SUPPORTED", False: "rejected ", None: "no answer"}[ok]
        print(f"{mark} {day} {kind:10} {claim[:88]}")
    flagged = [r for r in results if r[1] == "CORRECTION" and r[4] is not None and r[6] is not None]
    kept = [r for r in flagged if r[6]]
    if flagged:
        print(f"\ndetector-flagged messages with a hand label: {len(flagged)}, "
              f"of which real corrections {sum(bool(r[4]) for r in flagged)}")
        print(f"verifier kept {len(kept)}, of which real corrections {sum(bool(r[4]) for r in kept)}")
    mine = dict(db.execute("SELECT key, ok FROM reviews"))
    judged = [r for r in unique(results) if rkey(r) in mine]
    if judged:  # my verdicts replace the verifier's; leads I have not reviewed are left out
        agree = sum(bool(mine[rkey(r)]) == r[6] for r in judged)
        print(f"\nmy review covers {len(judged)} of {len(unique(results))} leads; "
              f"the verifier agrees with me on {agree} of {len(judged)}")
        rules = [r for r in judged if mine[rkey(r)]]
    else:
        print("\nno review of mine yet (python3 propose.py --review); falling back to the verifier")
        rules = [r for r in results if r[7]]
    if not rules:
        return print("no accepted lead, nothing to propose")
    old = open(path).read() if os.path.exists(path) else ""
    turn = ("CURRENT FILE:\n" + (scrub(old).strip() or "(empty)") + "\n\nCANDIDATES:\n" + "\n".join(
        f"{i}. " + (r[7] or f'Incident: {r[2]} The user wrote: "{r[3]}"') for i, r in enumerate(rules, 1)))
    proposal = merged(cached(db, model, MERGE, turn, key, max_tokens=8192), len(rules))
    if not proposal:
        return print("the merge step returned no usable line; run again")
    open("proposal.diff", "w").write(diff(old, [line for line, _ in proposal], TARGET))
    print(f"\n{len(rules)} accepted leads -> {len(proposal)} proposed lines, written to proposal.diff\n")
    for line, refs in proposal:
        print(line)
        for r in refs:
            day, kind, _, quote = rules[r - 1][:4]
            print(f'      from {day} {kind.lower()}: "{quote}"')
    used = {r for _, refs in proposal for r in refs}
    for i, r in enumerate(rules, 1):
        if i not in used:  # never drop an accepted lead silently
            print(f'left out by the merge ({r[0]} {r[1].lower()}): "{r[3]}"')


def check():
    summary = 'WORK: stuff "x y z w"\nFRICTION:\n- did wrong thing. "no, wrong file"\n- none\nPREFERENCES:\n- likes uv: “use uv always”'
    assert bullets(summary) == [("FRICTION", 'did wrong thing. "no, wrong file"', "no, wrong file"),
                                ("PREFERENCE", 'likes uv: "use uv always"', "use uv always")]
    lines = ["ASSISTANT: a", "ASSISTANT: no, wrong file", "USER: No,  wrong file please", "ASSISTANT: ok"]
    assert context(lines, "no, wrong file", before=1) == "   ASSISTANT: no, wrong file\n>> USER: No,  wrong file please\n   ASSISTANT: ok"
    assert context(lines, "never said") is None and context(lines, None) is None
    assert parse("VERDICT: SUPPORTED\nRULE: Ask before deleting.") == (True, "Ask before deleting.")
    assert parse("VERDICT: NOT SUPPORTED\nRULE: Ask first.") == (False, None)
    assert parse("VERDICT: SUPPORTED\nRULE: NONE") == (True, None) and parse("hmm") == (None, None)
    reply = "- Ask before deleting. [1, 3]\n- Invented rule.\n- Bad ref. [9]\nchatter"
    assert merged(reply, 3) == [("- Ask before deleting.", [1, 3])]
    d = diff("# Mine\n", ["- Ask before deleting."], "CLAUDE.md")
    assert "+- Ask before deleting." in d and "+" + HEADING in d and "-# Mine" not in d
    added = [x for x in diff("", ["- A."], "f").splitlines() if x.startswith("+") and not x.startswith("+++")]
    assert added == ["+" + HEADING, "+", "+- A."]  # no file yet: the proposal is the whole file
    a = ("d1", "FRICTION", "claim", "q", None, "KIND: FRICTION\nLINES:\n   ASSISTANT: x\n>> USER: no")
    b = ("d1", "CORRECTION", FLAGGED, "q", 1, "KIND: CORRECTION\nLINES:\n>> USER: no")
    c = ("d1", "CORRECTION", FLAGGED, "q", 0, "KIND: CORRECTION\nLINES:\n>> USER: other")
    assert marked(a) == ">> USER: no" and unique([a, b, c]) == [a, c] and rkey(a) != rkey(b)
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE checks (key PRIMARY KEY, answer)")
    db.execute("CREATE TABLE reviews (key PRIMARY KEY, ok)")
    stdin, sys.stdin = sys.stdin, io.StringIO("a\nr\n")
    with contextlib.redirect_stdout(io.StringIO()):
        review(db, "m", [a, c])
        review(db, "m", [a, c])  # everything reviewed: asks nothing, changes nothing
    sys.stdin = stdin
    assert dict(db.execute("SELECT key, ok FROM reviews")) == {rkey(a): 1, rkey(c): 0}
    print("check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        arg = lambda flag, default: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
        run(sqlite3.connect("coach.db"), arg("--model", DEFAULT_MODEL),
            os.path.expanduser(arg("--file", TARGET)), dry="--dry-run" in sys.argv,
            ask_me="--review" in sys.argv)
