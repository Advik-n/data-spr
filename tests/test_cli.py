import argparse
from procurement import baseline
from procurement.cli import execute, read_rows

def test_integration(tmp_path):
    import csv
    import pytest
    root = baseline.ROOT
    sample = read_rows(root/"sample_threads.csv")[:3]
    for name in ("threads.csv", "sample_threads.csv", "sample_answers.csv", "sample_submission.csv"):
        records = sample if "threads" in name else [
            r for r in read_rows(root/"sample_answers.csv") if r["thread_id"] in {s["thread_id"] for s in sample}
        ][::-1]
        with (tmp_path/name).open("w", newline="") as file:
            writer = csv.DictWriter(file, records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    args = argparse.Namespace(command="run", data=tmp_path, playbook=baseline.BOOK, run_dir=tmp_path/"run", output=None)
    assert execute(args)["completed"] == 3
    assert execute(args)["cache_hits"] == 3
    args.command = "evaluate"
    assert execute(args)["matches"] == 3
    args.command = "replay"
    assert execute(args)["rows"] == 3
    assert (args.run_dir/"submission.csv").read_bytes() == (args.run_dir/"replay.csv").read_bytes()
    args.command = "validate"
    assert execute(args)["rows"] == 3
    with (args.run_dir/"submission.csv").open("a") as file:
        file.write("extra,ACCEPT:ACCEPTABLE_PRICE:\n")
    with pytest.raises(ValueError):
        execute(args)




def test_review_routes_on_exact_outcome_disagreement():
    from unittest.mock import Mock
    from procurement.workflow import graph
    book = baseline.compile_policy()
    template = baseline.rows("sample_threads.csv")[0]
    thread = dict(template, quantity="1", **{"round": "1"})
    runtime = Mock()
    runtime.choose.return_value = ("2", {"method": "fixture"})
    flow = graph(book, "fixture", runtime)
    unit = baseline.resolve(thread, book)["Normalized unit"]
    thread["supplier_message"] = f"Original offer INR 1 per {unit}.\nUpdated offer INR 999999 per {unit}."
    decision = flow.invoke({"thread":thread})["trace"]
    assert runtime.choose.call_count == 1
    assert decision.price == 999999
    assert len(set(decision.review["candidate_labels"].values())) == 2
    runtime.choose.reset_mock()
    thread["supplier_message"] = thread["supplier_message"].replace("999999", "2")
    flow.invoke({"thread":thread})
    runtime.choose.assert_not_called()
