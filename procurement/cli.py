"""Command line driver for the offline procurement workflow."""
import argparse
import time
import hashlib
import json
import shutil
import sys
from tempfile import NamedTemporaryFile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from . import baseline
from .workflow import batch, digest, read_rows, replay, validate


def unpack(archive, destination, depth=0):
    if depth > 3:
        raise ValueError("Archive nesting limit exceeded")
    nested = []
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            relative = PurePosixPath(item.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in item.filename or ":" in item.filename:
                raise ValueError(f"Unsafe archive path {item.filename!r}")
            target = destination / relative
            if item.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            if target.suffix.lower() == ".zip":
                nested.append(target)
    for child in nested:
        unpack(child, child.with_suffix(".extracted"), depth+1)


@contextmanager
def discover(path, explicit_book=None):
    with TemporaryDirectory(prefix="procurement-data-") as tmp:
        root = path.resolve()
        if root.is_dir() and not list(root.rglob("threads.csv")):
            archives = sorted(root.glob("*.zip"))
            if len(archives) == 1:
                root = archives[0]
        if root.is_file():
            unpack(root, Path(tmp))
            root = Path(tmp)
        data = next((p.parent for p in sorted(root.rglob("threads.csv"), key=lambda p: (len(p.parts), p))
                     if all((p.parent/f).is_file() for f in ("sample_threads.csv","sample_submission.csv","sample_answers.csv"))), None)
        if data is None:
            raise ValueError(f"Cannot locate competition CSVs under {root}")
        if explicit_book:
            book = explicit_book.resolve()
        else:
            procedure = next(iter(sorted(root.rglob("policies/decision-procedure.md"))), None)
            if procedure is None:
                for i, archive in enumerate(sorted(root.rglob("playbook.zip"))):
                    dest = Path(tmp)/f"book-{i}"
                    unpack(archive, dest)
                    procedure = next(dest.rglob("policies/decision-procedure.md"), None)
                    if procedure:
                        break
            if procedure is None:
                raise ValueError("Cannot locate playbook")
            book = procedure.parents[1]
        yield data, book


def compile_book(book):
    fingerprint = digest([(str(p.relative_to(book)), hashlib.sha256(p.read_bytes()).hexdigest())
                          for p in sorted(book.rglob("*.md"))])
    return baseline.compile_policy(book), fingerprint


def write_report(directory, filename, report):
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf8", dir=directory, delete=False) as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    Path(stream.name).replace(directory / filename)


def execute(args):
    started = time.monotonic()
    profile = getattr(args, "profile", "baseline")
    if args.command == "replay":
        return {"rows": replay(args.run_dir, args.output or args.run_dir/"replay.csv")}
    with discover(args.data, args.playbook) as (data, book):
        ids = [r["thread_id"] for r in read_rows(data/"sample_submission.csv")]
        if args.command == "validate":
            return {"rows": validate(args.output or args.run_dir/"submission.csv", ids)}
        compiled, fingerprint = compile_book(book)
        if args.command == "compile-policy":
            return {"policy_hash": fingerprint, "categories": compiled[0], "suppliers": compiled[1], "amendments": compiled[2]}
        runtime = None
        if profile != "baseline" or args.command == "audit-policy":
            from .integration import ReviewRuntime
            args.run_dir.mkdir(parents=True, exist_ok=True)
            runtime = ReviewRuntime(profile, getattr(args, "transport", "local"), book, args.run_dir,
                                    getattr(args, "evaluated_model", []))
        if args.command == "audit-policy":
            from .providers import OperationalError, classify
            try:
                report = runtime.audit(book)
            except Exception as exc:
                report = {"errors": [str(exc) if isinstance(exc, OperationalError) else classify(exc)]}
            write_report(args.run_dir, "audit-policy.json", report)
            return report
        sample = args.command == "evaluate"
        rows = read_rows(data/("sample_threads.csv" if sample else "threads.csv"))
        if args.command == "inspect":
            if set(ids) != {r["thread_id"] for r in rows}:
                raise ValueError("Submission and thread ID sets differ")
            for row in rows:
                baseline.resolve(row, compiled)
            return {"threads": len(rows), "samples": len(read_rows(data/"sample_threads.csv")),
                    "categories": len({r["category"] for r in rows}), "suppliers": len({r["supplier"] for r in rows}),
                    "policy_hash": fingerprint}
        if sample:
            answers = {r["thread_id"]:r["label"] for r in read_rows(data/"sample_answers.csv")}
            ids = list(answers)
        labels, errors, hits = batch(rows, ids, compiled, fingerprint, args.run_dir, evaluation=sample, runtime=runtime)
        report = {"total": len(ids), "completed": len(labels), "cache_hits": hits, "errors": errors}
        if sample:
            misses = {i: {"expected": answers[i], "actual": labels.get(i)} for i in ids if answers[i] != labels.get(i)}
            slices = {}
            for field in ("category", "supplier"):
                groups = {}
                for row in rows:
                    group = groups.setdefault(row[field], {"matches": 0, "total": 0})
                    group["total"] += 1
                    group["matches"] += row["thread_id"] not in misses
                slices[field] = groups
            report.update(matches=len(ids)-len(misses), misses=misses, slices=slices)
        report.update(elapsed_seconds=round(time.monotonic()-started, 3), profile=profile, model_calls=runtime.registry.total if runtime else 0)
        write_report(args.run_dir, "evaluation.json" if sample else "run.json", report)
        return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["providers"]:
        from .providers import main as provider_main
        return provider_main(argv[1:])
    parser = argparse.ArgumentParser(prog="procurement")
    parser.add_argument("command", choices=["inspect","compile-policy","evaluate","run","validate","replay","audit-policy"])
    parser.add_argument("--data", type=Path, default=Path.cwd(), help="Competition directory or ZIP")
    parser.add_argument("--playbook", type=Path)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, help="CSV for validate or replay")
    parser.add_argument("--profile", choices=["baseline","assisted","reviewed"], default="baseline")
    parser.add_argument("--transport", choices=["local", "hosted"], default="local")
    parser.add_argument("--evaluated-model", action="append", default=[],
                        help="Explicit alias/model qualified by independent review evaluation")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir or Path("runs") / args.profile
    report = execute(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(report.get("errors") or report.get("misses")))


if __name__ == "__main__":
    raise SystemExit(main())
