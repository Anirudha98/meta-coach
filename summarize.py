"""Step 3: one short LLM summary per day, so a month of history fits in one context.

Each day's prompts and replies (secrets masked, long text clipped) go to the model. It
returns what was worked on, where the assistant caused friction, and standing preferences.
Friction and preference items must quote the user, and every quote is checked against
the source text, so a made-up item shows up as a quote that is not found.

Usage:  python3 summarize.py              # summarise new or changed days, print the digest
        python3 summarize.py --dry-run    # show what would be sent, send nothing
        python3 summarize.py --model <id> # another NVIDIA NIM model
        python3 summarize.py --check      # self-test, no network
Summaries are cached per day and model in coach.db; a day is redone when it gains messages.
"""
import re
import sqlite3
import sys
import zlib

from detect_llm import DEFAULT_MODEL, PASTE, api_key, ask, scrub
from signals import INJECTED, NOT_TYPED, classify

BUDGET = 48_000  # characters of transcript per request, roughly 12k tokens
# Hidden reasoning counts against max_tokens. Even at low effort a 24,000-character day used
# 3,200 tokens of it, and three days came back empty at 4,096.
MAX_TOKENS = 8192

SYSTEM = """You summarise one day of a user's conversations with an AI coding assistant.
The summary feeds an audit of how the assistant should be instructed, so friction matters most.
Write plain text in exactly this shape:

WORK: one or two sentences on what the user was getting done.
FRICTION:
- what the assistant got wrong, then the user's own words in "double quotes"
PREFERENCES:
- how the user wants to be worked with in later sessions, then their words in "double quotes"

Rules:
- List every friction item the day holds, one bullet each, at most 30 words before the
  quote. A busy day may have many bullets. A section with nothing is "- none".
- Friction means the assistant's fault: a wrong assumption, ignoring something already said,
  doing more or less than asked, or repeating a step that had failed. Before writing a
  bullet, find the ASSISTANT line that shows the mistake and the USER line that objects.
- Not friction: the user answering the assistant's question, the user asking for something
  new, the user's own slip (lines marked "USER ran in their own terminal" are the user's
  actions), or a tool, website or install failing by itself.
- A preference is something that should hold in later sessions: how to explain, what to ask
  before acting, facts about the user's setup. A requirement of today's task is not one.
- Each quote is at most 15 words, copied character for character from a "USER:" line, typos
  included. Never quote the assistant. Use double quotes for nothing else.
- Use only what the transcript shows. "[...cut...]" marks removed text; never guess at it."""
STAMP = zlib.crc32(SYSTEM.encode())  # a changed prompt makes cached summaries stale
BASH = re.compile(r"<bash-input>(.*?)</bash-input>", re.S)  # a command the user ran themself


def clip(text, n):
    """Collapse whitespace; keep the start and end of long text and mark the cut."""
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n // 2] + " [...cut...] " + text[-(n // 2):]


def transcripts(db):
    """{day: [line, ...]}. Days are local time; each session is kept together."""
    # ponytail: names of people are not masked, only key-, email- and number-shaped text
    # (see detect_llm.SECRETS). Upgrade path: a local model, so nothing leaves the machine.
    out, last = {}, None
    for day, sid, cwd, kind, text in db.execute(
            "SELECT date(ts,'localtime'), session_id, cwd, kind, text FROM messages "
            "WHERE kind IN ('prompt','reply') ORDER BY 1, session_id, ts, uuid, idx"):
        lines = out.setdefault(day, [])
        if (day, sid) != last:
            lines.append("=== session in folder: %s ===" % (cwd or "?").rstrip("/").split("/")[-1])
            last = (day, sid)
        ran = BASH.search(text)
        if kind == "reply":
            lines.append("ASSISTANT: " + clip(scrub(text), 600))
        elif ran:  # without this the model blamed the assistant for the user's own commands
            lines.append("USER ran in their own terminal: " + clip(scrub(ran.group(1)), 200))
        elif classify(text) == "interrupt":
            lines.append("USER: [interrupted the assistant]")
        elif classify(text) != "injected":
            typed = INJECTED.sub("", PASTE.sub("[PASTED TEXT]", text)).strip()
            if not typed.startswith(NOT_TYPED):
                lines.append("USER: " + clip(scrub(typed), 1500))
    return out


def parts(lines, budget):
    """Split lines into chunks of at most `budget` characters, never inside a line."""
    # ponytail: a day over the budget is stored as several summaries, not merged into one.
    # Add a merge call if days regularly need more than one request.
    out, size = [[]], 0
    for line in lines:
        if size + len(line) > budget and out[-1]:
            out.append([])
            size = 0
        out[-1].append(line)
        size += len(line) + 1
    return ["\n".join(chunk) for chunk in out if chunk]


def norm(text):
    return " ".join(text.lower().replace("’", "'").replace("“", '"').replace("”", '"').split())


def quotes_found(summary, source):
    """(found, total, missing): quoted snippets of the summary that occur in the source."""
    source, found, missing = norm(source), 0, []
    quotes = re.findall(r'"([^"]{4,})"', norm(summary))
    for quote in quotes:
        pieces = [p.strip() for p in re.split(r"\[\.\.\.cut\.\.\.\]|\.\.\.|…", quote) if len(p.strip()) > 3]
        if pieces and all(p in source for p in pieces):
            found += 1
        else:
            missing.append(quote)
    return found, len(quotes), missing


def mentions(digest_text, prompt):
    """True if any 4 words in a row that the user typed in `prompt` appear in the digest."""
    words = norm(scrub(re.sub(r"<!-- reply \d+ -->|^>.*$", "", prompt, flags=re.M))).split()
    return any(" ".join(words[i:i + 4]) in norm(digest_text) for i in range(max(1, len(words) - 3)))


def digest(db, model=DEFAULT_MODEL):
    """All summaries in day order as one text. This is what step 4 reads."""
    rows = db.execute("SELECT day, summary FROM summaries WHERE model=? ORDER BY day, part", (model,))
    return "\n\n".join(f"## {day}\n{summary}" for day, summary in rows)


def run(db, model, dry=False):
    db.execute("CREATE TABLE IF NOT EXISTS summaries "
               "(day, part, model, n, summary, PRIMARY KEY (day, part, model))")
    days = transcripts(db)
    key = None if dry else api_key()
    for day, lines in sorted(days.items()):
        chunks = parts(lines, BUDGET)
        if dry:
            print(f"{day}: {len(lines)} lines, {sum(map(len, chunks))} chars, {len(chunks)} request(s)")
            continue
        have = db.execute("SELECT count(*), max(n) FROM summaries WHERE day=? AND model=?",
                          (day, model)).fetchone()
        stamp = f"{len(lines)}:{STAMP}"
        if have == (len(chunks), stamp):
            continue  # cached; the day has not grown and the prompt has not changed
        db.execute("DELETE FROM summaries WHERE day=? AND model=?", (day, model))
        for i, chunk in enumerate(chunks):
            # At default effort the model spent all 4096 tokens reasoning about a 600-character
            # day and returned nothing. "low" answers in about 500.
            answer = ask(model, key, f"DAY {day}\n\n{chunk}", SYSTEM,
                         reasoning_effort="low", max_tokens=MAX_TOKENS)
            text = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()
            if not text:  # no usable answer: store nothing, so the next run asks again
                print(f"{day}: no answer", flush=True)
                continue
            db.execute("INSERT INTO summaries VALUES (?,?,?,?,?)", (day, i, model, stamp, text))
            db.commit()
            print(f"{day}: summarised", flush=True)
    sent = sum(len(line) for lines in days.values() for line in lines)
    if dry:
        print(f"total {sent} chars in {len(days)} days; nothing sent")
        return
    text = digest(db, model)
    print("\n" + text + "\n")
    found = total = 0
    for day, lines in sorted(days.items()):
        said = "\n".join(line for line in lines if line.startswith("USER: "))
        got = "\n".join(s for (s,) in db.execute(
            "SELECT summary FROM summaries WHERE day=? AND model=? ORDER BY part", (day, model)))
        f, t, missing = quotes_found(got, said)
        found, total = found + f, total + t
        for quote in missing:
            print(f"  quote not in source ({day}): {quote[:80]}")
    print(f"{model}: {sent} chars of history -> {len(text)} chars of summary "
          f"({sent / max(len(text), 1):.0f}x smaller)")
    print(f"quotes found in the user's own words: {found} of {total}")
    try:
        fixes = db.execute("SELECT m.text FROM labels l JOIN messages m USING (uuid, idx) "
                           "WHERE l.is_correction").fetchall()
        print(f"hand-labelled corrections quoted in the summaries: "
              f"{sum(mentions(text, t) for (t,) in fixes)} of {len(fixes)}")
    except sqlite3.OperationalError:
        pass  # no labels table yet


def check():
    assert clip("a  b\nc", 10) == "a b c" and clip("x" * 100, 10) == "xxxxx [...cut...] xxxxx"
    assert parts(["aaaa", "bbbb", "cc"], 9) == ["aaaa\nbbbb", "cc"] and parts([], 9) == []
    assert parts(["a" * 50], 9) == ["a" * 50]  # an oversized line still goes out, alone
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE messages (uuid, idx, session_id, ts, cwd, kind, tool, is_error, text)")
    rows = [("u1", "s1", "2026-01-01T10:00:00Z", "prompt", "fix it <system-reminder>hidden</system-reminder>"),
            ("u2", "s1", "2026-01-01T10:01:00Z", "reply", "done, mail a@b.com"),
            ("u3", "s1", "2026-01-01T10:02:00Z", "prompt", "<system-reminder>only this</system-reminder>"),
            ("u4", "s1", "2026-01-01T10:03:00Z", "prompt", "[Request interrupted by user]"),
            ("u5", "s1", "2026-01-01T10:04:00Z", "prompt",
             "<system-reminder>x</system-reminder>\n" + NOT_TYPED[1] + ": do the thing"),
            ("u6", "s2", "2026-01-01T10:05:00Z", "prompt",
             '<pasted_content id="1">private</pasted_content> no, wrong file'),
            ("u7", "s2", "2026-01-01T10:06:00Z", "prompt",
             "<bash-input>python3 x.py</bash-input><bash-stdout>out</bash-stdout><bash-stderr></bash-stderr>")]
    db.executemany("INSERT INTO messages VALUES (?,0,?,?,'/home/me/proj',?,NULL,0,?)", rows)
    (lines,) = transcripts(db).values()
    assert lines == ["=== session in folder: proj ===", "USER: fix it", "ASSISTANT: done, mail [EMAIL]",
                     "USER: [interrupted the assistant]", "=== session in folder: proj ===",
                     "USER: [PASTED TEXT] no, wrong file",
                     "USER ran in their own terminal: python3 x.py"], lines
    summary = 'FRICTION:\n- wrong file: "No, wrong   file"\n- invented: "you never listen"\n- "fix ... wrong file"'
    assert quotes_found(summary, "\n".join(lines)) == (2, 3, ["you never listen"])
    assert mentions('- he said "no, wrong file" twice', "<!-- reply 1 -->\n> quoted reply\n\nNo,  wrong file")
    assert not mentions("- none", "this was never summarised at all")
    print("check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else DEFAULT_MODEL
        run(sqlite3.connect("coach.db"), model, dry="--dry-run" in sys.argv)
