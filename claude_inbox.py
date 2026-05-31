"""
claude_inbox.py — interface to the Telegram→Claude file inbox.

monitor_train.py's /claude command appends messages to results/claude_inbox.jsonl
(no tmux needed).  The running Claude session polls this inbox, processes pending
messages, replies via tele.py, and marks them done here.

Usage
-----
  python claude_inbox.py list           # print PENDING messages (JSON lines)
  python claude_inbox.py done <ts>       # mark message with timestamp <ts> done
  python claude_inbox.py all             # print all (incl. done) for audit
"""
import sys, os, json

INBOX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results", "claude_inbox.jsonl")


def _load():
    if not os.path.exists(INBOX):
        return []
    out = []
    with open(INBOX, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out


def _save(recs):
    with open(INBOX, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    recs = _load()
    if cmd == "list":
        pend = [r for r in recs if not r.get("done")]
        if not pend:
            print("(no pending messages)")
        for r in pend:
            print(json.dumps({"ts": r.get("ts"), "msg": r.get("msg")},
                             ensure_ascii=False))
    elif cmd == "all":
        for r in recs:
            print(json.dumps(r, ensure_ascii=False))
    elif cmd == "done":
        if len(sys.argv) < 3:
            print("usage: claude_inbox.py done <ts>"); return
        ts = sys.argv[2]
        n = 0
        for r in recs:
            if str(r.get("ts")) == str(ts) and not r.get("done"):
                r["done"] = True; n += 1
        _save(recs)
        print(f"marked {n} done")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
