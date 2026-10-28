# AGENTS.md

ZAC 2026 LifeMate dataset repo. No application build/test setup at root; assets are under `data/train+public_test/`.

## Repo shape

- Splits: `train/` (1545 labeled samples), `public_test/` (634 unlabeled), `private_test/` released later.
- Each split: `audio/<id>/`, `<split>.jsonl`, `<split>_tools_catalog.json`, `<split>_environment/<id>/state.json`, optional `sample_submission.jsonl`, `zaic_tools/`.
- Train and public_test have different app sets and separate `zaic_tools` versions (train: 46 apps/147 tools; public_test: 36 apps/122 tools). Always use the matching split's code/catalog.
- Not a git repository. Ignore `__MACOSX/` metadata.

## Critical data-access rule

- Never read, copy, or parse `<split>_environment/` or `state.json`.
- All app state access must go through that split's `zaic_tools.ToolEnv`.
- Create exactly one `ToolEnv` per sample, once. Re-creating it resets state and breaks scoring.
- Every call, including schema-rejected calls, is logged by `env.get_tool_calls()` and must appear in submission.

## Commands

Run from `data/train+public_test`:

```sh
python -m train.zaic_tools --split train --root . --apps
python -m public_test.zaic_tools --split public_test --root . --apps
python -m <split>.zaic_tools --split <split> --root . --id <id> --tool <app.tool> --args '{}'
python -m <split>.zaic_tools --split <split> --root . --validate submission.jsonl
```

Inference pattern: `ROOT = Path(".")`, `SPLIT = "..."`, `sys.path.insert(0, str(ROOT / SPLIT))`, import `ToolEnv`, write UTF-8 JSONL with `id`, `tool_calls`, `state_hash`.

## Runtime quirks

- `zaic_tools` is stdlib-only Python 3.10+.
- In this checkout, pass `--root .` from `data/train+public_test` (or `root=...`); default root resolution assumes released package layout.
- Schema errors are the only hard failures (`{"ok": false, "error": ...}`). Missing IDs, empty searches, or insufficient balance return `{"ok": true, "done": false, ...}` or `[]` and do not change state.
- `env.validate(tool, args)` checks schema without executing.
- State hash excludes `_meta`, live apps (`weather`, `gold_price`, `fuel_price`), and free-text keys (`address`, `body`, `content`, etc.). Fuzzy `x-match` fields score by token-F1 > 0.6.
- `sys.ask_missing_info` uses `tool_name` and `params` (list); it is logged and scored like any call.
- Out-of-scope requests submit `[]`.
- Installed apps vary per sample; allowed calls are installed app tools plus `sys.ask_missing_info`.

## Competition constraints

- Offline inference, no external APIs.
- Total model size < 6B; LLM < 5B with context <= 16384 tokens.
- 30 seconds per sample.
- Deterministic decoding and fixed seeds.
- Do not read or label public/private test data.
- Final Docker runs with data mounted at `/data` and produces `result/submission.jsonl`.

## Key docs

- `docs/description.md`, `docs/data.md`, `docs/evaluation.md`, `docs/rule.md`, `docs/guideline.md`
- `data/train+public_test/README.md`
- `data/train+public_test/{train,public_test}/zaic_tools/README.md`