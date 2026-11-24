import csv
import io
import zipfile

import pytest

from procurement.cli import unpack
from procurement.workflow import validate


def test_nested_playbook(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("playbook/policies/decision-procedure.md", "rules")
    outer = tmp_path / "desk.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("playbook.zip", inner.getvalue())
    destination = tmp_path / "extract"
    unpack(outer, destination)
    assert (destination / "playbook.extracted/playbook/policies/decision-procedure.md").read_text() == "rules"


@pytest.mark.parametrize("member", ["../escape", "/absolute", "C:/escape", "foo\\bar"])
def test_unsafe_member(tmp_path, member):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, "payload")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        unpack(path, tmp_path / "out")


@pytest.mark.parametrize("label", ["COUNTER:COUNTER_OFFER:-1", "COUNTER:COUNTER_OFFER:20.0",
                                   "ACCEPT:ACCEPTABLE_PRICE", "ACCEPT:ABOVE_WALKAWAY:",
                                   "COUNTER:COUNTER_OFFER:1,000"])
def test_label_contract(tmp_path, label):
    path = tmp_path / "submission.csv"
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows([("thread_id","label"), ("T1",label)])
    with pytest.raises(ValueError, match="Invalid submission label"):
        validate(path, ["T1"])
