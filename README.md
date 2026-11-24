# The Procurement Desk

An offline, policy-driven Kaggle solution for predicting the complete negotiation label from an email thread. The verified candidate covers 2,000 threads, 15 categories and 40 suppliers. The deterministic engine matched all 40 official worked labels. Hidden accuracy is unknown.

## Reproduce the submission

Use Python 3.12 and uv. Run these commands from the repository root.

```bash
uv lock --check
uv sync --locked
uv run --locked procurement inspect --data the-procurement-desk.zip
uv run --locked procurement compile-policy
uv run --locked pytest -q
uv run --locked procurement evaluate --run-dir runs/best
uv run --locked procurement run --run-dir runs/best
uv run --locked procurement validate --run-dir runs/best
uv run --locked procurement replay --run-dir runs/best --output /tmp/procurement-replay.csv
cmp runs/best/submission.csv /tmp/procurement-replay.csv
```

The input loader accepts the competition ZIP or a directory containing the four CSVs and playbook. Use `--data PATH` and `--playbook PATH` to override discovery. A run directory holds a SQLite decision store and reports. Repeating a command resumes matching rows. Replay uses the saved ordered snapshot. The validated file for manual Kaggle upload is [submission.csv](runs/best/submission.csv).

## Evidence and evaluation

The engine distinguishes the buyer's counter from the supplier's previous price. In T00007, a 5% discount on ₹3,910 gives ₹3,714.50. Equality to the concession or approval limit is acceptable. The serializer rounds only a counter price. The graph uses the same Python rules during local evaluation and specialist review.

- [Worked evaluation](runs/best/evaluation.json) measures exact matches on the supplied labels by thread ID.
- [Coverage report](runs/best/coverage.json) counts message families and policy coverage separately for submission and worked rows. Regenerate it with `uv run --locked python -m procurement.profile_data`.
- [Policy fixtures](tests/test_policy.py) and [offer fixtures](tests/test_offer_mechanics.py) exercise independent policy and numerical boundaries.
- [Architecture](docs/architecture.md) describes source precedence, evidence handling, routing, and operational limits.

## Optional model assistance

Copy `.env.example` to `.env` and paste keys on your machine. Git ignores `.env`. Baseline runs and the test suite require no API key. These commands list configured routes, probe a local Ollama server, and evaluate each optional profile.

```bash
uv run --locked procurement providers list
uv run --locked procurement providers probe --provider ollama --profile local
uv run --locked procurement evaluate --profile assisted
uv run --locked procurement evaluate --profile reviewed
```

To select hosted inference explicitly, use `--transport hosted` for a run or evaluation, or `--profile hosted` with the provider probe. Set a named model in the provider's environment setting for reproducibility. The reviewed profile accepts `--evaluated-model ALIAS/MODEL` for a route that you have qualified on independent evidence cases. A passing tool probe gates DeepAgents delegation. A policy audit requires a reachable tool-capable model.

```bash
uv run --locked procurement audit-policy --transport local --run-dir runs/audit
```

The default free-only configuration requires confirmation of an enforced no-overage account limit for account-dependent routes. Enter those aliases in `LLM_CONFIRMED_NO_CHARGE_PROVIDERS`. OpenRouter checks the live model catalog before selecting a free route. Availability and tool support can change. [OpenRouter model catalog](https://openrouter.ai/api/v1/models)
