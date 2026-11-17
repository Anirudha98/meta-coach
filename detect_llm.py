"""Correction detector that asks an LLM, scored against the hand labels.

Unlike the keyword rules in signals.py, the model sees the assistant's previous reply
next to the user's message. Zero-shot: the prompt holds a definition and no examples
from the labelled data, so scoring on all labels is fair.

Usage:  export NVIDIA_API_KEY=nvapi-...   (or put that line in .env)
        python3 detect_llm.py                      # predict + score, default model
        python3 detect_llm.py --model google/gemma-4-31b-it
        python3 detect_llm.py --check              # self-test, no network
Every prompt I typed is scored (compare.py needs that); the ones with a hand label are
used to measure the detector. Predictions are cached per model in coach.db.
"""
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

from label import score
from signals import typed

URL = "https://integrate.api.nvidia.com/v1/chat/completions"  # OpenAI-compatible
DEFAULT_MODEL = "mistralai/mistral-large-4-1t-a52b"
# Reasoning models spend ~500 tokens thinking before the one-word answer. At 512 the
# answer was cut off for 18 of 142 prompts, so leave real headroom.
MAX_TOKENS = 4096

SYSTEM = """You label messages a user sent to an AI coding assistant.
A CORRECTION is the user telling the assistant that its last step was wrong, unwanted,
based on a wrong assumption, or ignored something the user already said.
Everything else is OTHER: a new request, a follow-up question, an answer to the
assistant's question, approval, a progress check, or a report of an outside error.
Reply with exactly one word: CORRECTION or OTHER."""

# ponytail: pattern scrubbing only. Catches key-shaped strings, emails and phone numbers;
# a password typed as an ordinary word still gets through. Upgrade path: a secret scanner
# such as detect-secrets, or run a local model so nothing leaves the machine.
SECRETS = [
    (re.compile(r"\d{6}(-\d{6}){3,}"), "[KEY]"),                                  # BitLocker
    (re.compile(r"\b(sk|nvapi|ghp|gho|github_pat|xox[bp])[-_][\w-]{16,}"), "[KEY]"),
    (re.compile(r"[A-Za-z0-9+/_-]{40,}"), "[KEY]"),                               # opaque tokens
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\+?\d[\d ]{8,}\d"), "[NUMBER]"),
]
PASTE = re.compile(r"<pasted_content[^>]*>.*?</pasted_content[^>]*>", re.S)


def scrub(text):
    for pattern, repl in SECRETS:
        text = pattern.sub(repl, text)
    return text


def build(reply, prompt):
    """The user turn sent to the model. Long text is cut and marked, never silently."""
    reply, prompt = scrub(reply or ""), scrub(PASTE.sub("[PASTED TEXT]", prompt))
    if len(reply) > 800:
        reply = "[...earlier part cut...] " + reply[-800:]
    if len(prompt) > 1500:
        prompt = prompt[:1500] + " [...rest cut...]"
    return f"ASSISTANT'S PREVIOUS REPLY:\n{reply or '(none, this is the first message)'}\n\nUSER'S MESSAGE:\n{prompt}"


def parse(answer):
    """True / False from the model's reply; None if it said neither word."""
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).upper()
    if "CORRECTION" in answer:
        return True
    return False if "OTHER" in answer else None


def ask(model, key, user_turn, system=SYSTEM, **extra):
    """One chat completion. `extra` adds request fields, e.g. reasoning_effort="low"."""
    body = json.dumps(dict({"model": model, "temperature": 0, "max_tokens": MAX_TOKENS,
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user_turn}]}, **extra)).encode()
    req = urllib.request.Request(URL, body, {"Authorization": "Bearer " + key,
                                             "Content-Type": "application/json"})
    for wait in (2, 8, 30, None):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or wait is None:
                sys.exit(f"API error {e.code}: {e.read().decode()[:300]}")
            time.sleep(wait)  # rate limited or server hiccup: back off and retry
        except OSError as e:  # timeout or dropped connection
            if wait is None:
                sys.exit(f"Network error: {e}")
            time.sleep(wait)


def api_key(name="NVIDIA_API_KEY"):
    key = os.environ.get(name)
    if not key and os.path.exists(".env"):
        for line in open(".env"):
            if line.startswith(name + "="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    return key or sys.exit(f"Set {name} (export it, or add it to .env)")


def run(db, model):
    key = api_key()
    db.execute("CREATE TABLE IF NOT EXISTS llm_preds "
               "(uuid, idx, model, is_correction, PRIMARY KEY (uuid, idx, model))")
    rows = db.execute("SELECT uuid, idx, session_id, kind, text FROM messages "
                      "WHERE kind IN ('prompt','reply') ORDER BY session_id, ts, uuid, idx").fetchall()
    labels = {(u, i): y for u, i, y in db.execute("SELECT uuid, idx, is_correction FROM labels")}
    done = {(u, i) for u, i in db.execute("SELECT uuid, idx FROM llm_preds WHERE model=?", (model,))}
    last_reply = {}
    for uuid, idx, sid, kind, text in rows:
        if kind == "reply":
            last_reply[sid] = text
        elif typed(text) and (uuid, idx) not in done:  # every prompt I typed, labelled or not
            pred = parse(ask(model, key, build(last_reply.get(sid), text)))
            if pred is None:  # no usable answer: don't cache it, so the next run asks again
                print("?", end="", flush=True)
                continue
            db.execute("INSERT INTO llm_preds VALUES (?,?,?,?)", (uuid, idx, model, pred))
            db.commit()
            print(".", end="", flush=True)
    got = db.execute("SELECT p.is_correction, l.is_correction, m.text FROM llm_preds p "
                     "JOIN labels l USING (uuid, idx) JOIN messages m USING (uuid, idx) "
                     "WHERE p.model=?", (model,)).fetchall()
    p, r, tp, fp, fn = score([(bool(p), bool(y)) for p, y, _ in got])
    total, flagged = db.execute("SELECT count(*), sum(is_correction) FROM llm_preds WHERE model=?", (model,)).fetchone()
    print(f"\n{model}: {total} prompts scored, {flagged or 0} flagged as corrections")
    print(f"against my hand labels: {len(got)} of {len(labels)} labelled prompts answered")
    print(f"precision {p:.2f}  recall {r:.2f}  (hits {tp}, false alarms {fp}, missed {fn})")
    for pred, y, text in got:
        if bool(pred) != bool(y):
            print("  false alarm:" if pred else "  missed:     ", " ".join(scrub(text).split())[:90])


def check():
    assert scrub("key 111111-222222-333333-444444 ok") == "key [KEY] ok"
    assert scrub("mail a.b@x.com call +91 12345 67890") == "mail [EMAIL] call [NUMBER]"
    assert scrub("nvapi-" + "a" * 20) == "[KEY]" and scrub("fix the login page") == "fix the login page"
    assert parse("CORRECTION") is True and parse(" other.") is False and parse("maybe") is None
    assert parse("<think>is it a CORRECTION?</think>OTHER") is False
    turn = build("word " * 400, "<pasted_content id=\"1\">secret</pasted_content> no, wrong file")
    assert "secret" not in turn and "[PASTED TEXT] no, wrong file" in turn and "cut" in turn
    print("check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else DEFAULT_MODEL
        run(sqlite3.connect("coach.db"), model)
