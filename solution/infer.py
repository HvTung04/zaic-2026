import argparse
import json
import sys
from pathlib import Path

ROOT_DEFAULT = Path("data/train+public_test")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solution.agent import run_sample
from solution.asr import ASR
from solution.llm import LLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT_DEFAULT))
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--output", default="submission.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-gold-text", action="store_true")
    ap.add_argument("--asr-model", default="small")
    args = ap.parse_args()

    root = Path(args.root)
    split = args.split
    in_path = root / split / f"{split}.jsonl"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    asr = None if args.use_gold_text else ASR(args.asr_model)
    llm = LLM()

    n = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            calls, h = run_sample(item, split, root, asr, llm, use_gold_text=args.use_gold_text)
            fout.write(json.dumps({"id": item["id"], "tool_calls": calls, "state_hash": h}, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            print(f"{n}: {item['id']}", flush=True)
            if args.limit and n >= args.limit:
                break
    print(f"wrote {n} lines to {out_path}")


if __name__ == "__main__":
    main()
