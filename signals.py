"""Step 2: friction signals per session, computed in plain code (no LLM).

Usage:  python3 signals.py            # reads coach.db, writes table session_signals
        python3 signals.py --check    # self-test
"""
import re
import sqlite3
import sys

# Blocks the app injects into the user slot: <system-reminder>..., <bash-input>..., etc.
INJECTED = re.compile(r"<([a-z][a-z-]*)\b[^>]*>.*?</\1>", re.S)
INTERRUPT = re.compile(r"^\[Request interrupted")

# ponytail: keyword heuristics, tuned on 13 sessions of one user's English. They miss
# polite corrections and misfire on e.g. "do it again for file B". Upgrade path: label
# ~200 prompts by hand, measure precision/recall, then swap in a small classifier.
CORRECTION = re.compile(
    r"^(no|nope|wrong)\b|\b(i (told|said|meant)|not what i|that'?s (not|wrong)"
    r"|already (told|said|did|compared|gave)|y again|why again|still (not|the same|wrong)"
    r"|not working|did ?n[o']?t work|does ?n[o']?t work|undo|revert)\b", re.I)
NUDGE = re.compile(r"\b(how much (is )?done|are (you|u) (done|stuck)|any update|status\?)", re.I)


def classify(text):
    """Label one user-slot message: injected | interrupt | correction | nudge | prompt."""
    if INTERRUPT.match(text.strip()):
        return "interrupt"
    typed = INJECTED.sub("", text).strip()
    if not typed:
        return "injected"
    if NUDGE.search(typed):
        return "nudge"
    if CORRECTION.search(typed):
        return "correction"
    return "prompt"


def session_signals(db):
    """One dict per session with raw counts and two rates."""
    out = []
    for (sid,) in db.execute("SELECT DISTINCT session_id FROM messages ORDER BY 1"):
        msgs = db.execute("SELECT kind, tool, is_error, text FROM messages "
                          "WHERE session_id=? ORDER BY ts, uuid, idx", (sid,)).fetchall()
        labels = [classify(t) for k, _, _, t in msgs if k == "prompt"]
        calls = [(tool, t) for k, tool, _, t in msgs if k == "tool_use"]
        turns = sum(l != "injected" for l in labels)  # everything the user actually sent
        s = {
            "session_id": sid,
            "turns": turns,
            "corrections": labels.count("correction"),
            "nudges": labels.count("nudge"),
            "interrupts": labels.count("interrupt"),
            "tool_calls": len(calls),
            "tool_errors": sum(e for k, _, e, _ in msgs if k == "tool_result"),
            "repeat_calls": len(calls) - len(set(calls)),  # identical tool + input again
        }
        friction = s["corrections"] + s["nudges"] + s["interrupts"]
        s["friction_rate"] = round(friction / turns, 3) if turns else 0.0
        s["tool_error_rate"] = round(s["tool_errors"] / len(calls), 3) if calls else 0.0
        out.append(s)
    return out


def save(db, rows):
    cols = list(rows[0])
    db.execute("DROP TABLE IF EXISTS session_signals")
    db.execute("CREATE TABLE session_signals (%s)" % ", ".join(cols))
    db.executemany("INSERT INTO session_signals VALUES (%s)" % ",".join("?" * len(cols)),
                   [tuple(r.values()) for r in rows])
    db.commit()


def check():
    cases = {
        "<system-reminder>\nstuff\n</system-reminder>": "injected",
        "<system-reminder>x</system-reminder>\nfix the login page": "prompt",
        "<bash-input>ls</bash-input><bash-stdout>a</bash-stdout>": "injected",
        "[Request interrupted by user]": "interrupt",
        "how much done ?": "nudge",
        "u already compared once y again>": "correction",
        "no, use the other file": "correction",
        "it is still not working": "correction",
        "now can i use Instagram now ?": "prompt",
        "what is the speed of ssd ?": "prompt",
    }
    for text, want in cases.items():
        assert classify(text) == want, (text, classify(text), want)

    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE messages (uuid, idx, session_id, ts, cwd, kind, tool, is_error, text)")
    m = [("a", "prompt", None, 0, "build it"), ("b", "tool_use", "Bash", 0, "ls"),
         ("c", "tool_result", None, 1, "boom"), ("d", "tool_use", "Bash", 0, "ls"),
         ("e", "prompt", None, 0, "no, wrong folder"), ("f", "prompt", None, 0, "<x-y>z</x-y>")]
    db.executemany("INSERT INTO messages VALUES (?,0,'s','t','/',?,?,?,?)", m)
    (s,) = session_signals(db)
    assert (s["turns"], s["corrections"], s["tool_errors"], s["repeat_calls"]) == (2, 1, 1, 1), s
    assert s["friction_rate"] == 0.5 and s["tool_error_rate"] == 0.5, s
    print("check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        db = sqlite3.connect("coach.db")
        rows = session_signals(db)
        save(db, rows)
        cols = [c for c in rows[0] if c != "session_id"]
        print("session   " + " ".join(f"{c[:11]:>11}" for c in cols))
        for r in rows:
            print(r["session_id"][:8] + "  " + " ".join(f"{r[c]:>11}" for c in cols))
