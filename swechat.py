"""Test the correction detector on SWE-chat, a public set of real coding-agent sessions.

SWE-chat (Stanford, huggingface.co/datasets/SALT-NLP/SWE-chat) labels every user prompt as
correction, rejection, failure_report or non_pushback. Those labels were made by a model,
and "failure_report" is something my own labels do not count as a correction, so results
are shown per label as well as as one score.

The dataset is gated: accept its terms on the dataset page, then put a read token in .env
as HF_TOKEN=... Rows are read through Hugging Face's rows API, 100 at a time from random
places in the conversations table, so nothing large is downloaded.

Usage:  python3 swechat.py --peek         # column names and a few rows, to check assumptions
        python3 swechat.py --fetch 300    # sample about 300 labelled prompts into coach.db
        python3 swechat.py                # run the detector on the sample and score it
        python3 swechat.py --check        # self-test, no network
"""
import http.client
import json
import random
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from detect_llm import DEFAULT_MODEL, api_key, ask, build, parse
from label import score

API = "https://datasets-server.huggingface.co/rows"
BASE = {"dataset": "SALT-NLP/SWE-chat", "config": "conversations", "split": "train"}
MINE = ("correction", "rejection")  # the SWE-chat labels that match what I call a correction
LABELS = ("correction", "rejection", "failure_report", "non_pushback")


def rows(token, offset, length=100):
    url = API + "?" + urllib.parse.urlencode(dict(BASE, offset=offset, length=length))
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token,
                                               "User-Agent": "dataset-client/1.0"})
    for wait in (3, 10, 30, 60, 120, 300, None):  # the rows API rate-limits after ~60 calls
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                sys.exit(f"Hugging Face says {e.code}: accept the dataset's terms on its page, "
                         "and check HF_TOKEN in .env is a read token for the same account.")
            if e.code not in (429, 500, 502, 503) or wait is None:
                sys.exit(f"Hugging Face API error {e.code}: {e.read().decode()[:300]}")
            time.sleep(wait)
        except (OSError, http.client.HTTPException) as e:  # timeouts, resets, cut-off bodies
            if wait is None:
                sys.exit(f"Network error: {e}")
            time.sleep(wait)


def pairs(batch):
    """[(turn_id, session_id, reply, prompt, label)] from one window of consecutive rows.

    Rows are stored in conversation order (checked with --peek on 2026-11-17). Only
    turn_type "assistant_response" counts as the reply; "assistant_thinking" rows share the
    assistant role but are hidden reasoning. A prompt whose
    previous reply is outside the window, or cut short by the API, is skipped: the detector
    needs the end of that reply.
    """
    said, out = {}, []
    for item in batch:
        row, cut = item["row"], item.get("truncated_cells") or []
        sid, text = row.get("session_id"), row.get("content") or ""
        if row.get("turn_type") == "user_prompt":
            if row.get("prompt_pushback") in LABELS and sid in said and "content" not in cut:
                out.append((row["turn_id"], sid, said[sid], text, row["prompt_pushback"]))
        elif row.get("turn_type") == "assistant_response" and text.strip():  # not thinking or tools
            said[sid] = None if "content" in cut else text
        if said.get(sid, "") is None:
            said.pop(sid)
    return out


def fetch(db, token, n, seed=0):
    # ponytail: windows are picked by table position, so long sessions are over-represented.
    # Sample session ids from the sessions table first if that bias matters.
    db.execute("CREATE TABLE IF NOT EXISTS swechat (turn_id PRIMARY KEY, session_id, reply, prompt, label)")
    have = db.execute("SELECT count(*) FROM swechat").fetchone()[0]
    total, rng = rows(token, 0, 1)["num_rows_total"], random.Random(seed + have)  # resume: new windows
    for window in range(n * 3):  # about 1 prompt per window (tool rows fill the rest)
        have = db.execute("SELECT count(*) FROM swechat").fetchone()[0]
        if have >= n:
            break
        time.sleep(2)  # stay under the rate limit
        found = pairs(rows(token, rng.randrange(total - 100))["rows"])
        db.executemany("INSERT OR IGNORE INTO swechat VALUES (?,?,?,?,?)", found)
        db.commit()
        print(f"\rwindow {window + 1}: {have + len(found)} prompts", end="", flush=True)
    print("\n" + ", ".join(f"{label} {count}" for label, count in db.execute(
        "SELECT label, count(*) FROM swechat GROUP BY 1 ORDER BY 2 DESC")))


def table(got):
    """Text report from [(label, flagged)]: the flag rate per label, then two overall scores."""
    out = []
    for label in LABELS:
        flags = [f for l, f in got if l == label]
        if flags:
            out.append(f"  {label:15} {len(flags):5} prompts, flagged {sum(flags):4} ({100 * sum(flags) / len(flags):.0f}%)")
    for name, positive in (("corrections and rejections only", MINE), ("any pushback", LABELS[:3])):
        p, r, tp, fp, fn = score([(bool(f), l in positive) for l, f in got])
        out.append(f"{name}: precision {p:.2f}  recall {r:.2f}  (hits {tp}, false alarms {fp}, missed {fn})")
    return "\n".join(out)


def run(db, model):
    key = api_key()
    db.execute("CREATE TABLE IF NOT EXISTS swechat_preds (turn_id, model, is_correction, PRIMARY KEY (turn_id, model))")
    todo = db.execute("SELECT turn_id, reply, prompt FROM swechat WHERE turn_id NOT IN "
                      "(SELECT turn_id FROM swechat_preds WHERE model=?)", (model,)).fetchall()
    with ThreadPoolExecutor(4) as pool:  # sqlite is written from this thread only
        answers = pool.map(lambda t: parse(ask(model, key, build(t[1], t[2]))), todo)
        for (turn_id, _, _), pred in zip(todo, answers):
            if pred is not None:  # no usable answer: not stored, so the next run asks again
                db.execute("INSERT INTO swechat_preds VALUES (?,?,?)", (turn_id, model, pred))
                db.commit()
            print("?" if pred is None else ".", end="", flush=True)
    got = db.execute("SELECT s.label, p.is_correction FROM swechat s JOIN swechat_preds p "
                     "USING (turn_id) WHERE p.model=?", (model,)).fetchall()
    fetched = db.execute("SELECT count(*) FROM swechat").fetchone()[0]
    print(f"\n{model} on SWE-chat: {len(got)} of {fetched} fetched prompts scored\n" + table(got) if got
          else "\nnothing fetched yet: python3 swechat.py --fetch 300")


def peek(token):
    d = rows(token, 0, 40)
    print("rows in table:", d["num_rows_total"])
    print("columns:", ", ".join(f"{f['name']}" for f in d["features"]))
    for item in d["rows"]:
        r = item["row"]
        print(f"{str(r.get('session_id'))[:8]} {str(r.get('turn_id'))[:10]:10} {str(r.get('role')):9} "
              f"{str(r.get('turn_type')):16} tool={str(r.get('tool_name'))[:10]:10} "
              f"conv={r.get('is_conversational')} push={r.get('prompt_pushback')} | "
              f"{' '.join(str(r.get('content') or '').split())[:60]}")


def check():
    def row(sid, turn_type, role, content, tool=None, label=None, tid="t", cut=()):
        return {"row": {"session_id": sid, "turn_id": tid, "turn_type": turn_type, "role": role, "content": content,
                        "tool_name": tool, "prompt_pushback": label}, "truncated_cells": list(cut)}
    batch = [row("a", "user_prompt", "user", "orphan: no reply seen yet", label="correction", tid="1"),
             row("a", "assistant_response", "assistant", "I renamed the file."),
             row("a", "assistant_thinking", "assistant", "hidden reasoning, not a reply"),
             row("a", "tool_use", "assistant", "ls -la", tool="Bash"),
             row("b", "assistant_response", "assistant", "other session"),
             row("a", "user_prompt", "user", "no, the other file", label="correction", tid="2"),
             row("a", "user_prompt", "user", "unlabelled", tid="3"),
             row("a", "assistant_response", "assistant", "cut reply", cut=["content"]),
             row("a", "user_prompt", "user", "after a cut reply", label="rejection", tid="4"),
             row("b", "user_prompt", "user", "thanks", label="non_pushback", tid="5")]
    assert pairs(batch) == [("2", "a", "I renamed the file.", "no, the other file", "correction"),
                            ("5", "b", "other session", "thanks", "non_pushback")], pairs(batch)
    report = table([("correction", 1), ("correction", 0), ("failure_report", 1), ("non_pushback", 0)])
    assert "correction          2 prompts, flagged    1 (50%)" in report, report
    assert "corrections and rejections only: precision 0.50  recall 0.50" in report
    assert "any pushback: precision 1.00  recall 0.67" in report
    print("check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif "--peek" in sys.argv:
        peek(api_key("HF_TOKEN"))
    elif "--fetch" in sys.argv:
        fetch(sqlite3.connect("coach.db"), api_key("HF_TOKEN"), int(sys.argv[sys.argv.index("--fetch") + 1]))
    else:
        model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else DEFAULT_MODEL
        run(sqlite3.connect("coach.db"), model)
