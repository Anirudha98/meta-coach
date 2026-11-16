"""Step 1: load Claude Code session logs into SQLite.

Usage:  python3 load_logs.py            # reads ~/.claude/projects, writes coach.db
        python3 load_logs.py --check    # self-test on a fake log
Safe to re-run: rows already loaded are skipped.
"""
import glob
import json
import os
import sqlite3
import sys

LOGS = os.path.expanduser("~/.claude/projects/*/*.jsonl")
SCHEMA = """CREATE TABLE IF NOT EXISTS messages (
    uuid TEXT, idx INTEGER,          -- one log line can hold several blocks
    session_id TEXT, ts TEXT, cwd TEXT,
    kind TEXT,                       -- prompt | reply | tool_use | tool_result
    tool TEXT, is_error INTEGER, text TEXT,
    PRIMARY KEY (uuid, idx))"""


def rows(line):
    """Turn one log line into zero or more message rows."""
    try:
        d = json.loads(line)
    except ValueError:
        return
    role = d.get("type")
    if role not in ("user", "assistant") or d.get("isSidechain") or d.get("isMeta"):
        return  # skip bookkeeping lines, subagent chatter, injected context
    content = d.get("message", {}).get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    for i, b in enumerate(content or []):
        t, kind, tool, err, text = b.get("type"), None, None, 0, ""
        if t == "text":
            kind, text = ("prompt" if role == "user" else "reply"), b.get("text", "")
        elif t == "tool_use":
            kind, tool, text = "tool_use", b.get("name"), json.dumps(b.get("input"))
        elif t == "tool_result":
            kind, err = "tool_result", int(bool(b.get("is_error")))
            c = b.get("content")
            text = c if isinstance(c, str) else json.dumps(c)
        if kind:  # thinking and images are dropped
            yield (d["uuid"], i, d.get("sessionId"), d.get("timestamp"), d.get("cwd"),
                   kind, tool, err, text)


def load(db_path, pattern=LOGS):
    db = sqlite3.connect(db_path)
    db.execute(SCHEMA)
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                db.executemany("INSERT OR IGNORE INTO messages VALUES (?,?,?,?,?,?,?,?,?)",
                               rows(line))
    db.commit()
    return db


def check():
    import tempfile
    d = tempfile.mkdtemp()
    base = {"sessionId": "s1", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/x"}
    lines = [
        dict(base, type="user", uuid="a", message={"content": "fix the bug"}),
        dict(base, type="assistant", uuid="b", message={"content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]}),
        dict(base, type="user", uuid="c", message={"content": [
            {"type": "tool_result", "is_error": True, "content": "boom"}]}),
        dict(base, type="user", uuid="d", isMeta=True, message={"content": "injected"}),
        dict(base, type="system", uuid="e"),
    ]
    with open(os.path.join(d, "s.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(l) for l in lines) + "\nnot json\n")
    for _ in range(2):  # second run must not duplicate
        db = load(os.path.join(d, "t.db"), os.path.join(d, "*.jsonl"))
    got = db.execute("SELECT kind, tool, is_error FROM messages ORDER BY uuid").fetchall()
    assert got == [("prompt", None, 0), ("tool_use", "Bash", 0), ("tool_result", None, 1)], got
    print("check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        db = load("coach.db")
        for kind, n in db.execute("SELECT kind, COUNT(*) FROM messages GROUP BY kind"):
            print(f"{kind:12} {n}")
        print("sessions    ", db.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()[0])
