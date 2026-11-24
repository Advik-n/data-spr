import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from procurement.audit import audit_policy
from procurement.review import Budget, BudgetExceeded, Finding, Snapshot, review_evidence


class ScriptModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, ident):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(ident)}])


def test_real_deepagent_supervisor_and_delegates(tmp_path):
    doc = "policies/decision-procedure.md"
    (tmp_path/"policies").mkdir()
    (tmp_path/doc).write_text("Accept equality.")
    proposal = dict(interpretation="Equality passes", reproducing_case="p=L=1",
                    expected_outcome="ACCEPT:ACCEPTABLE_PRICE:",
                    citations=[dict(document=doc, first=1, last=1, excerpt="Accept equality.")])
    script = []
    for name in ["policy-amendment", "precedent-analysis", "regression-design"]:
        script.extend([call("task", dict(description="Check equality", subagent_type=name), len(script)),
                       call("Proposal", proposal, len(script)+1)])
    script.append(AIMessage(content="Delegates agree on equality."))
    report = audit_policy(ScriptModel(responses=script), tmp_path)
    assert len(report["proposals"]) == 3
    assert report["model_calls"] == 7


def test_read_only_tools_and_bounds():
    snapshot = Snapshot({"precedents/P.md": "Use the higher price."}, {"supplier": "INR 10"})
    lookup, search, span, policy, outcomes = snapshot.tools()
    assert search.invoke({"query": "higher"}) == ["precedents/P.md"]
    assert span.invoke(dict(message_id="supplier", start=4, end=6)) == "10"
    with pytest.raises(KeyError):
        lookup.invoke(dict(document="../secret", first=1, last=1))
    with pytest.raises(ValueError):
        span.invoke(dict(message_id="supplier", start=0, end=30))


def test_schema_reviewer_validates_evidence_and_run_budget():
    snapshot = Snapshot({}, {"supplier": "INR 10"})
    class Reviewer:
        def with_structured_output(self, schema):
            assert schema is Finding
            return self
        def invoke(self, messages):
            return dict(state="confirmed", fact="quote", evidence=[
                dict(message_id="supplier", start=4, end=6, excerpt="10")])
    model = Reviewer()
    budget = Budget(1, 0)
    result = review_evidence(model, snapshot, "Which quote?", "quote-reference", budget)
    assert result.evidence[0].excerpt == "10"
    with pytest.raises(BudgetExceeded):
        review_evidence(model, snapshot, "Which quote?", "quote-reference", budget)
    with pytest.raises(ValueError):
        Snapshot({}, {"supplier": "INR 11"}).validate(result)


def test_supervisor_cannot_invoke_default_general_purpose_delegate():
    model = ScriptModel(responses=[call("task", {"description": "Read the host",
                                                 "subagent_type": "general-purpose"}, 0)])
    with pytest.raises(ValueError, match="outside investigation scope"):
        review_evidence(model, Snapshot({}), "Resolve quote", "quote-reference", Budget(3, 3),
                        tool_probe_passed=True, evaluated_route=True)


def test_budget_caps_delegate_model_calls():
    model = ScriptModel(responses=[call("task", {"description": "Find quote",
                                                 "subagent_type": "quote-reference"}, 0)])
    with pytest.raises(BudgetExceeded):
        review_evidence(model, Snapshot({}), "Resolve quote", "quote-reference", Budget(1, 2),
                        tool_probe_passed=True, evaluated_route=True)
