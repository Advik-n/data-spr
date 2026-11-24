"""Review-only policy audit with source-validated proposals."""
import json
from pathlib import Path

from pydantic import BaseModel, Field

from .review import Budget, Snapshot, investigate


class Citation(BaseModel):
    document: str
    first: int = Field(ge=1)
    last: int = Field(ge=1)
    excerpt: str = Field(min_length=1)


class Proposal(BaseModel):
    interpretation: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1, max_length=12)
    reproducing_case: str = Field(min_length=1)
    expected_outcome: str = Field(min_length=1)


def audit_policy(model, playbook, *, max_calls=16, run_budget=None):
    root = Path(playbook).resolve(strict=True)
    documents = {}
    for folder in ("policies", "suppliers", "categories", "amendments", "precedents"):
        for path in sorted((root / folder).glob("*.md")):
            if path.resolve().is_relative_to(root) and not path.is_symlink():
                documents[path.relative_to(root).as_posix()] = path.read_text(encoding="utf8")
    if "policies/decision-procedure.md" not in documents:
        raise ValueError("Playbook root must contain policies/decision-procedure.md")
    snapshot = Snapshot(documents)
    roles = {"policy-amendment": "Investigate date, field, and supplier precedence.",
             "precedent-analysis": "Investigate a precedent against its governing rules.",
             "regression-design": "Design a minimal boundary regression with an expected exact label."}

    def validate(record):
        for ref in record.citations:
            lines = documents[ref.document].splitlines()
            if (not 1 <= ref.first <= ref.last <= len(lines)
                    or "\n".join(lines[ref.first-1:ref.last]) != ref.excerpt):
                raise ValueError("Invalid policy citation")
        return record

    budget = Budget(max_calls, max_calls*4)
    budgets = (budget, run_budget) if run_budget is not None else (budget,)
    records, reconciliation = investigate(
        model, snapshot.tools()[:2], roles,
        "Audit the playbook. Give each delegate the relevant logical IDs. Cite exact complete lines without line-number prefixes. Available documents: " + json.dumps(sorted(documents)),
        Proposal, budgets, validate,
    )
    return {"status": "pending_code_review", "proposals": {k:v.model_dump() for k,v in records.items()},
            "reconciliation": reconciliation, "model_calls": max_calls-budget.remaining["model"]}


def main(argv=None):
    """Explicit invocation only. A cached or fresh capability probe gates delegation."""
    import argparse
    from .providers import Registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playbook", type=Path, required=True)
    parser.add_argument("--profile", choices=("local", "hosted"), required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe-cache", type=Path, required=True)
    args = parser.parse_args(argv)
    registry = Registry(args.profile)
    provider = registry.providers[args.provider]
    if args.model not in registry.models(provider):
        parser.error("Model is unavailable under the selected billing and profile settings")
    reports = registry.probe([args.provider], args.probe_cache)
    tested = next((r for r in reports if r.get("model") == args.model), {})
    if not tested.get("checks", {}).get("tool", {}).get("passed"):
        parser.error("A passing tool-call probe is required for this model")
    limit = min(int(registry.env["LLM_MAX_POLICY_AUDIT_CALLS"]),
                int(registry.env["LLM_MAX_CALLS_PER_RUN"]) - registry.total)
    report = audit_policy(registry.client(provider, args.model), args.playbook, max_calls=limit)
    print(json.dumps({"provider": args.provider, "model": args.model, **report}, indent=2))


if __name__ == "__main__":
    main()
