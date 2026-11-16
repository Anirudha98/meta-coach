"""Hand-label prompts as correction / not, then score the detector against the labels.

Usage:  python3 label.py            # label; c = correction, n = not, q = quit (resumes later)
        python3 label.py --score    # precision / recall of signals.classify
        python3 label.py --check    # self-test of the scoring maths

A correction = you telling the assistant its last step was wrong, unwanted, or ignored
what you said. A new request or a follow-up question is NOT a correction.
"""
import sqlite3
import sys

from signals import classify

ORDER = "ORDER BY session_id, ts, uuid, idx"


def score(pairs):
    """pairs = [(predicted_is_correction, labelled_is_correction)] -> precision, recall, counts."""
    tp = sum(p and y for p, y in pairs)
    fp = sum(p and not y for p, y in pairs)
    fn = sum(y and not p for p, y in pairs)
    return (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0, tp, fp, fn)


def label(db):
    db.execute("CREATE TABLE IF NOT EXISTS labels (uuid, idx, is_correction, PRIMARY KEY (uuid, idx))")
    rows = db.execute("SELECT uuid, idx, session_id, kind, text FROM messages "
                      "WHERE kind IN ('prompt','reply') " + ORDER).fetchall()
    done = set(db.execute("SELECT uuid, idx FROM labels"))
    todo = [r for r in rows if r[3] == "prompt" and (r[0], r[1]) not in done
            and classify(r[4]) not in ("injected", "interrupt")]
    last_reply = {}
    for n, (uuid, idx, sid, kind, text) in enumerate(rows):
        if kind == "reply":
            last_reply[sid] = text
            continue
        if (uuid, idx) in done or classify(text) in ("injected", "interrupt"):
            continue
        left = sum(1 for r in todo if (r[0], r[1]) not in done)
        print("\n" + "=" * 70 + f"\n{left} left\n")
        print("ASSISTANT SAID: ..." + " ".join(last_reply.get(sid, "(nothing yet)").split())[-300:])
        print("\nYOU SAID:       " + " ".join(text.split())[:600])
        while True:
            a = input("\ncorrection? [c / n / q] ").strip().lower()
            if a in ("c", "n", "q"):
                break
        if a == "q":
            break
        db.execute("INSERT INTO labels VALUES (?,?,?)", (uuid, idx, int(a == "c")))
        db.commit()
        done.add((uuid, idx))
    print(f"\n{len(done)} labelled. Run: python3 label.py --score")


def report(db):
    rows = db.execute("SELECT m.text, l.is_correction FROM labels l "
                      "JOIN messages m USING (uuid, idx)").fetchall()
    if not rows:
        sys.exit("no labels yet - run python3 label.py first")
    p, r, tp, fp, fn = score([(classify(t) == "correction", bool(y)) for t, y in rows])
    print(f"labelled {len(rows)}, of which corrections {tp + fn}")
    print(f"precision {p:.2f}  recall {r:.2f}  (hits {tp}, false alarms {fp}, missed {fn})")
    for t, y in rows:
        if y and classify(t) != "correction":
            print("  missed:", " ".join(t.split())[:100])


def check():
    assert score([(True, True), (True, False), (False, True), (False, False)]) == (0.5, 0.5, 1, 1, 1)
    assert score([(False, False)]) == (0.0, 0.0, 0, 0, 0)
    print("check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        db = sqlite3.connect("coach.db")
        report(db) if "--score" in sys.argv else label(db)
