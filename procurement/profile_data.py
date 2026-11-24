"""Report family coverage, public-label slices, and archive integrity."""
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from .baseline import BOOK, ROOT, compile_policy, evaluate, resolve, rows

FAMILIES = {
    "earlier_and_latest": r"\[Latest message", "buyer_counter": r"\[Earlier message — Acme",
    "correction": r"CORRECTION —", "historical_discount": r"off our previous quote",
    "multiple_currency_amounts": r"(?:₹|Rs\.|INR|USD|GBP|EUR).*?(?:₹|Rs\.|INR|USD|GBP|EUR)",
    "range": r"between .* and ", "tier": r"orders of up to", "total": r"in total for all",
    "list_discount": r"discount on our list", "volume_condition": r"off for orders of",
    "period": r"per (?:year|month)", "foreign_currency": r"[$€£]|\b(?:USD|GBP|EUR|JPY|AED|SGD)\b",
    "gst_inclusive": r"incl.*?GST|GST included", "free_units": r"free .* for every",
    "negated_contract": r"no automatic renewal|not asking for exclusivity",
    "reviewer_directed": r"bot screens|language model|automated system|reviewing assistant|AI AGENT|AI सहायक|admin override|assistant:|previous instructions",
    "word_price": r" rupees", "pack": r"per pack", "ream": r"per ream",
    "per_100": r"per 100 ", "dozen": r"per dozen", "scaled_numeric": r"\d(?:k\b| lakh)",
}


def archive_inventory(blob, depth=0):
    if depth > 3:
        raise ValueError("Archive nesting too deep")
    result = {}
    with zipfile.ZipFile(blob) as archive:
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename or ":" in item.filename:
                raise ValueError(item.filename)
            if item.is_dir():
                continue
            if item.filename in result:
                raise ValueError("Duplicate member")
            data = archive.read(item)
            result[item.filename] = hashlib.sha256(data).hexdigest()
            if item.filename.endswith(".zip"):
                result[item.filename] = archive_inventory(io.BytesIO(data), depth+1)
    return result


def main():
    book = compile_policy()
    samples = rows("sample_threads.csv")
    full = rows("threads.csv")
    labels = {r["thread_id"]:r["label"] for r in rows("sample_answers.csv")}
    sample_ids = [r["thread_id"] for r in samples]
    assert len(labels)==len(set(sample_ids))==len(samples) and labels.keys()==set(sample_ids)
    inventory = archive_inventory(ROOT/"the-procurement-desk.zip")
    for relative, digest in inventory.items():
        assert hashlib.sha256((ROOT/"the-procurement-desk"/relative).read_bytes()).hexdigest()==digest
        if relative.endswith(".csv"):
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    reports = {}
    for key, dataset in [("submission",full), ("worked",samples)]:
        families, categories, suppliers, matched = Counter(), Counter(), Counter(), Counter()
        for row in dataset:
            resolve(row,book)
            categories[row["category"]] += 1
            suppliers[row["supplier"]] += 1
            good = evaluate(row,book)["label"] == labels.get(row["thread_id"])
            for name, pattern in FAMILIES.items():
                if re.search(pattern,row["supplier_message"],re.I):
                    families[name] += 1
                    if good:
                        matched[name] += 1
        reports[key] = {"categories":categories,"suppliers":suppliers,
                        "families":{k:families[k] for k in FAMILIES}}
        if key=="worked":
            reports[key]["family_exact_matches"]=matched
    reports["archive_members_verified"] = len(inventory)
    reports["documents"] = len(list(BOOK.rglob("*.md")))
    out = ROOT/"runs/best/coverage.json"
    out.write_text(json.dumps(reports,indent=2,sort_keys=True)+"\n")

if __name__ == "__main__":
    main()
