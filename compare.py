"""Step 5: friction before and after the instructions file was applied.

A session is "after" if it started after the cutoff, because instructions are read when a
session starts. The same instruments run on both sides:

  flagged    messages the LLM detector calls corrections. Noisy (precision 0.42 on my
             labels), but it is the same noise on both sides.
  confirmed  mistakes I accepted in `propose.py --review`. Trustworthy, needs my review.
  interrupts and failed tool calls, from signals.py, no LLM.

Usage:  python3 compare.py                        # the table, and whether the gap means anything
        python3 compare.py --cutoff 2026-11-17T06:44:58Z
        python3 compare.py --check                # self-test
Before running: load_logs.py, then detect_llm.py so new prompts are scored.
"""
import sqlite3
import sys
from math import comb

from detect_llm import DEFAULT_MODEL
from signals import session_signals, typed

CUTOFF = "2026-11-17T06:44:58Z"  # when ~/.claude/CLAUDE.md was created, UTC
MIN_PROMPTS = 100  # with fewer after-prompts than this, the after rate is mostly noise


def fisher(a, b, c, d):
    """Two-sided Fisher exact p-value for the table [[a, b], [c, d]]."""
    n, row, col = a + b + c + d, a + b, a + c
    p = lambda x: comb(col, x) * comb(n - col, row - x) / comb(n, row)
    return min(1.0, sum(p(x) for x in range(max(0, row + col - n), min(row, col) + 1)
                        if p(x) <= p(a) * (1 + 1e-9)))


def needed(hits, n, factor=0.5, limit=5000):
    """Fewest after-prompts at which `factor` times the before rate gives p < 0.05, else None."""
    for m in range(20, limit + 1, 20) if n else ():
        k = round(m * hits / n * factor)
        if fisher(hits, n - hits, k, m - k) < 0.05:
            return m
    return None


def periods(db, cutoff, model=DEFAULT_MODEL):
    """Raw counts for {"before": ..., "after": ...}."""
    start = dict(db.execute("SELECT session_id, min(ts) FROM messages GROUP BY 1"))
    out = {side: dict.fromkeys(("sessions", "prompts", "unscored", "flagged", "interrupts",
                                "tool_calls", "tool_errors"), 0) for side in ("before", "after")}
    side = lambda sid: out["after" if start[sid] > cutoff else "before"]
    for s in session_signals(db):
        side(s["session_id"])["sessions"] += 1
        for k in ("interrupts", "tool_calls", "tool_errors"):
            side(s["session_id"])[k] += s[k]
    for sid, text, flag in db.execute(
            "SELECT m.session_id, m.text, p.is_correction FROM messages m LEFT JOIN llm_preds p "
            "ON p.uuid = m.uuid AND p.idx = m.idx AND p.model = ? WHERE m.kind = 'prompt'", (model,)):
        if typed(text):
            side(sid)["unscored" if flag is None else "prompts"] += 1
            side(sid)["flagged"] += flag or 0
    return out


def confirmed(db, cutoff, model=DEFAULT_MODEL):
    """{side: (mistakes I accepted, leads not reviewed yet)} from propose.py's review table."""
    # ponytail: leads carry a day, not a session, so a lead from the cutoff day counts as
    # "before". Carry the session id through propose.gather if that day ever matters.
    from propose import gather, rkey, unique
    day = db.execute("SELECT date(?, 'localtime')", (cutoff,)).fetchone()[0]
    mine = dict(db.execute("SELECT key, ok FROM reviews"))
    out = {"before": [0, 0], "after": [0, 0]}
    for lead in unique(gather(db, model)):
        if lead[1] != "PREFERENCE":
            side = out["after" if lead[0] > day else "before"]
            side[0] += bool(mine.get(rkey(lead)))
            side[1] += rkey(lead) not in mine
    return out


def report(db, cutoff, model=DEFAULT_MODEL):
    p, c = periods(db, cutoff, model), confirmed(db, cutoff, model)
    b, a = p["before"], p["after"]
    rate = lambda hits, n: f"{hits} ({100 * hits / n:.1f} per 100)" if n else "-"
    seen = lambda side: rate(c[side][0], p[side]["prompts"]) + (f" +{c[side][1]} unreviewed" if c[side][1] else "")
    rows = [("sessions", b["sessions"], a["sessions"]),
            ("prompts scored", b["prompts"], a["prompts"]),
            ("flagged by detector", rate(b["flagged"], b["prompts"]), rate(a["flagged"], a["prompts"])),
            ("mistakes I confirmed", seen("before"), seen("after")),
            ("interrupts", b["interrupts"], a["interrupts"]),
            ("failed tool calls", rate(b["tool_errors"], b["tool_calls"]), rate(a["tool_errors"], a["tool_calls"]))]
    print(f"cutoff {cutoff}\n{'':22}{'before':>32}{'after':>20}")
    for name, x, y in rows:
        print(f"{name:22}{str(x):>32}{str(y):>20}")
    if b["unscored"] + a["unscored"]:
        print(f"\n{b['unscored'] + a['unscored']} prompts not scored yet: run python3 detect_llm.py")
    if a["prompts"] < MIN_PROMPTS:
        need = needed(b["flagged"], b["prompts"])
        print(f"\nToo early to compare: {a['prompts']} scored prompts after the cutoff, the floor is {MIN_PROMPTS}.")
        print(f"A halved flagged rate would need about {need} after-prompts before it could be told from chance (p < 0.05)."
              if need else "With this baseline, even a halved flagged rate could not be told from chance.")
    else:
        pv = fisher(b["flagged"], b["prompts"] - b["flagged"], a["flagged"], a["prompts"] - a["flagged"])
        print(f"\nflagged rate, before vs after: p = {pv:.3f} (Fisher exact). "
              + ("Unlikely to be chance." if pv < 0.05 else "Could easily be chance."))
    print("This is not an experiment: the tasks differ between the two periods, so a gap shows direction, not cause.")


def check():
    assert abs(fisher(3, 0, 0, 3) - 0.1) < 1e-9 and abs(fisher(1, 9, 11, 3) - 0.002759) < 1e-5
    assert fisher(5, 5, 5, 5) == 1.0
    m = needed(12, 142)
    assert m and fisher(12, 130, round(m * 12 / 142 / 2), m - round(m * 12 / 142 / 2)) < 0.05
    assert needed(0, 100) is None and needed(5, 0) is None
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE messages (uuid, idx, session_id, ts, cwd, kind, tool, is_error, text)")
    db.execute("CREATE TABLE llm_preds (uuid, idx, model, is_correction)")
    rows = [("a", "old", "2026-01-01T10:00:00Z", "prompt", 0, "build it"),
            ("b", "old", "2026-01-01T10:01:00Z", "prompt", 0, "no, wrong folder"),
            ("c", "old", "2026-01-01T10:02:00Z", "prompt", 0, "<system-reminder>x</system-reminder>"),
            ("d", "old", "2026-01-03T10:00:00Z", "prompt", 0, "still the old session"),  # started before
            ("e", "new", "2026-01-02T10:00:00Z", "prompt", 0, "fix it"),
            ("f", "new", "2026-01-02T10:01:00Z", "tool_use", 0, "ls"),
            ("g", "new", "2026-01-02T10:02:00Z", "tool_result", 1, "boom")]
    db.executemany("INSERT INTO messages VALUES (?,0,?,?,'/',?,'Bash',?,?)", rows)
    db.executemany("INSERT INTO llm_preds VALUES (?,0,'m',?)", [("a", 0), ("b", 1), ("e", 0)])
    p = periods(db, "2026-01-02T00:00:00Z", "m")
    assert (p["before"]["sessions"], p["before"]["prompts"], p["before"]["flagged"], p["before"]["unscored"]) == (1, 2, 1, 1), p
    assert (p["after"]["sessions"], p["after"]["prompts"], p["after"]["tool_errors"]) == (1, 1, 1), p
    print("check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        arg = lambda flag, default: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
        report(sqlite3.connect("coach.db"), arg("--cutoff", CUTOFF), arg("--model", DEFAULT_MODEL))
