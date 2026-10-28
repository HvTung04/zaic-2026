import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    gold = {r["id"]: r for r in load_jsonl(args.gold)}
    # gold train labels are inside train.jsonl, not submission format
    preds = {r["id"]: r for r in load_jsonl(args.pred)}

    state_ok = 0
    exact_calls = 0
    total = 0
    for sid, p in preds.items():
        total += 1
        g = gold.get(sid)
        if g is None:
            continue
        # gold may be train row with label
        if "label" in g:
            gold_hash = None
            # compute gold hash by replaying? Need external; skip hash compare here
            gold_calls = g["label"]["tool_calls"]
        else:
            gold_hash = g.get("state_hash")
            gold_calls = g.get("tool_calls", [])
        if gold_hash is not None and p.get("state_hash") == gold_hash:
            state_ok += 1
        if p.get("tool_calls") == gold_calls:
            exact_calls += 1
    print(f"samples={total} exact_calls={exact_calls} state_hash_match={state_ok}")


if __name__ == "__main__":
    main()
