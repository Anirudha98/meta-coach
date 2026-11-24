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
        python3 swechat.py --judge        # three other models label the sample; majority vote
        python3 swechat.py --check        # self-test, no network
"""
import http.client
import json
import random
import re
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


# Stand-in for hand labels: three models, none of them the detector's, with a stricter prompt
# than the detector's. Majority vote; a tie or no answer is "unsure". GLM 5.3 and gpt-oss-20b
# were tried and did not answer on 2026-11-17, so two of the three are NVIDIA's.
JUDGES = ("nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-super-vl-120b-a12b", "google/gemma-4-31b-it")
VERDICTS = ("INJECTED", "BORDERLINE", "CORRECTION", "OTHER")
JUDGE = """You label messages sent to an AI coding assistant, to build a test set.
Read the assistant's previous reply and the message after it, then pick ONE label:
INJECTED: not typed by a person. Tool or command output, system notices, hook text,
  auto-generated summaries of earlier context, file contents pasted with no request.
CORRECTION: the person says the assistant's last step was wrong or unwanted, rested on a
  wrong assumption, or ignored something the person had already said.
  e.g. "no, I meant the other file", "why did you delete that?", "I told you not to use npm".
BORDERLINE: a redirect that could be a correction or a new request, e.g. "actually, use
  Postgres instead" with no sign the earlier choice was a mistake.
OTHER: everything else: a new task, a follow-up question, approval, an answer to the
  assistant's question, or a report that something outside failed ("the build fails with X")
  without saying the assistant did it wrong.
Reply with the label only: INJECTED, CORRECTION, BORDERLINE or OTHER."""


def verdict(answer):
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).upper()
    found = [v for v in VERDICTS if v in answer]
    return found[0].lower() if len(found) == 1 else None  # two labels named: no answer


# Second answer key with one model per maker, so two NVIDIA models cannot outvote Gemma.
PAIR = ("nvidia/nemotron-3-ultra-550b-a55b", "google/gemma-4-31b-it")


def vote(labels, n):
    """Majority of n judges; a tie or too few answers is "unsure"."""
    labels = [l for l in labels if l]
    top = max(set(labels), key=labels.count) if labels else None
    return top if top and labels.count(top) * 2 > n else "unsure"


def votes(db, judges):
    got = {}
    for turn_id, model, label in db.execute("SELECT turn_id, model, label FROM swechat_judge"):
        if model in judges:
            got.setdefault(turn_id, []).append(label)
    return {t: vote(l, len(judges)) for t, l in got.items()}


def judge(db):
    key = api_key()
    db.execute("CREATE TABLE IF NOT EXISTS swechat_judge (turn_id, model, label, PRIMARY KEY (turn_id, model))")
    for model in JUDGES:
        todo = db.execute("SELECT turn_id, reply, prompt FROM swechat WHERE turn_id NOT IN "
                          "(SELECT turn_id FROM swechat_judge WHERE model=?)", (model,)).fetchall()
        print(f"{model}: {len(todo)} to label ", end="", flush=True)
        with ThreadPoolExecutor(4) as pool:
            answers = pool.map(lambda t: verdict(ask(model, key, build(t[1], t[2]), JUDGE)), todo)
            for (turn_id, _, _), label in zip(todo, answers):
                if label:
                    db.execute("INSERT INTO swechat_judge VALUES (?,?,?)", (turn_id, model, label))
                    db.commit()
                print("?" if label is None else ".", end="", flush=True)
        print()
    for judges in (JUDGES, PAIR):
        print(judge_report(db, judges))


def judge_report(db, judges):
    v = votes(db, judges)
    counts = {}
    for label in v.values():
        counts[label] = counts.get(label, 0) + 1
    out = [f"\n=== vote of {len(judges)}: " + ", ".join(j.split("/")[1] for j in judges),
           "labels: " + ", ".join(f"{l} {n}" for l, n in sorted(counts.items(), key=lambda x: -x[1]))]
    mine = [(m, v[t]) for t, m in db.execute("SELECT turn_id, label FROM swechat_mine") if t in v]
    if mine:  # how far the vote can be trusted: agreement with the 100 I labelled by hand
        same = sum(a == b for a, b in mine)
        out.append(f"vs my hand labels: same label on {same} of {len(mine)}")
        for want in ("correction", "borderline", "injected", "other"):
            said = [b for a, b in mine if a == want]
            out.append(f"  my {want:10} {len(said):3}: vote said " + ", ".join(
                f"{l} {said.count(l)}" for l in sorted(set(said))))
        p, r, tp, fp, fn = score([(b == "correction", a == "correction") for a, b in mine if b != "unsure"])
        out.append(f"  vote as a correction finder (unsure left out): precision {p:.2f}  recall {r:.2f}")
    preds = dict(db.execute("SELECT turn_id, is_correction FROM swechat_preds WHERE model=?", (DEFAULT_MODEL,)))
    det = [(lab, preds[t]) for t, lab in v.items() if t in preds and lab in ("correction", "borderline", "other")]
    for name, pos in (("borderline as not", ("correction",)), ("borderline as correction", ("correction", "borderline"))):
        p, r, tp, fp, fn = score([(bool(f), lab in pos) for lab, f in det])
        out.append(f"detector vs this vote, {len(det)} typed prompts, {name}: precision {p:.2f}  recall {r:.2f}"
                   f"  (hits {tp}, false alarms {fp}, missed {fn})")
    return "\n".join(out)


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
    assert verdict("<think>maybe OTHER</think> Correction.") == "correction"
    assert verdict("CORRECTION or OTHER") is None and verdict("") is None
    assert vote(["correction", "correction", None], 3) == "correction"
    assert vote(["correction", "other", "borderline"], 3) == "unsure" and vote([None, None, "other"], 3) == "unsure"
    assert vote(["other", "other"], 2) == "other" and vote(["other", "correction"], 2) == "unsure"
    print("check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif "--peek" in sys.argv:
        peek(api_key("HF_TOKEN"))
    elif "--votes" in sys.argv:  # reports only, no API calls
        print("\n".join(judge_report(sqlite3.connect("coach.db"), j) for j in (JUDGES, PAIR)))
    elif "--judge" in sys.argv:
        judge(sqlite3.connect("coach.db"))
    elif "--fetch" in sys.argv:
        fetch(sqlite3.connect("coach.db"), api_key("HF_TOKEN"), int(sys.argv[sys.argv.index("--fetch") + 1]))
    else:
        model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else DEFAULT_MODEL
        run(sqlite3.connect("coach.db"), model)
