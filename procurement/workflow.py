"""Offline LangGraph inference and content-addressed replay."""
import csv
import hashlib
import json
import re
import sqlite3
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import baseline
from .models import ClauseFinding, ConversionStep, DecisionTrace, EffectivePolicy, Evidence, Message, NormalizedOffer, QuoteCandidate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class State(TypedDict, total=False):
    thread: dict[str, str]
    policy: EffectivePolicy
    messages: tuple[Message, ...]
    offer: NormalizedOffer
    trace: DecisionTrace
    options: dict
    error: str


def split_messages(text):
    markers = list(re.finditer(r"^\[(Earlier|Latest) message[^\]\n]*\]", text, re.M))
    boundaries = [(m.end(), m[0]) for m in markers] or [(0, "")]
    for i, (start, header) in enumerate(boundaries):
        end = markers[i+1].start() if i+1 < len(markers) else len(text)
        block = Message(message_id=f"m{i+1}", speaker="buyer" if "— Acme Procurement" in header else "supplier",
                        start=start, end=end, excerpt=text[start:end])
        block.verify(text)
        yield block


def graph(book, policy_hash, runtime=None):
    def resolve(state):
        fields = baseline.resolve(state["thread"], book)
        return {"policy": EffectivePolicy(fields=fields, policy_hash=policy_hash)}

    def split(state):
        return {"messages": tuple(split_messages(state["thread"]["supplier_message"]))}

    def extract(state):
        thread = state["thread"]
        try:
            offer = explain_offer(thread, state["policy"].fields, state["messages"])
        except (KeyError, IndexError, ValueError, StopIteration, ArithmeticError) as exc:
            return {"error": f"extract: {type(exc).__name__}: {exc}"}
        return {"offer": offer}

    def evaluate(state):
        security, contract = clause_findings(state["messages"])
        trace = decide(state["thread"], state["policy"], state["offer"], security, contract)
        return {"trace": trace.model_copy(update={"messages": state["messages"]})}

    def alternatives(state):
        # A second, independently stated current unit quote can change the decision.
        # Single-line range, tier, correction and bundle mechanics belong to extract.
        block = state["messages"][-1]
        thread = state["thread"]
        options = {}
        lines = list(re.finditer(r"[^\n]+", block.excerpt))
        quotes = [line for line in lines if baseline.MONEY.search(line[0])
                  and re.search(r"\bper\b", line[0], re.I)]
        if len(quotes) < 2:
            return {"options": {}}
        if len(quotes) > 4:
            return {"error": "too_many_quote_alternatives"}
        trace = state["trace"]
        for i, line in enumerate(quotes):
            # Normalize the selected line with the ordinary parser. Evaluate clauses
            # against the original thread using the precedence branch already computed.
            single = dict(thread, supplier_message=line[0])
            price, _ = baseline.extract(single, state["policy"].fields)
            if trace.price is None or price is None:
                return {"error": "unresolved_quote_basis"}
            result = baseline.decide(thread, state["policy"].fields, price,
                trace.label == "ESCALATE:PROMPT_INJECTION:", trace.label == "ESCALATE:BANNED_TERM:")
            options[str(i+1)] = {"result": result, "span": {"message_id": block.message_id,
                "start": line.start(), "end": line.end(), "excerpt": line[0]}}
        return {"options": options}

    def review(state):
        try:
            choice, meta = runtime.choose(state["thread"], state["policy"].fields,
                                          state["messages"], state["options"])
        except Exception as exc:
            from .providers import OperationalError, classify
            return {"error": str(exc) if isinstance(exc, OperationalError) else classify(exc)}
        option = state["options"][choice]
        offer = explain_offer(state["thread"], state["policy"].fields, state["messages"], option["span"])
        security, contract = clause_findings(state["messages"])
        trace = decide(state["thread"], state["policy"], offer, security, contract)
        if trace.label != option["result"]["label"]:
            raise ValueError("Selected option disagrees with replayed evidence")
        metadata = dict(meta, candidate_labels={k:v["result"]["label"] for k,v in state["options"].items()})
        return {"offer": offer, "trace": trace.model_copy(update={"messages": state["messages"], "review": metadata})}

    builder = StateGraph(State)
    for name, fn in [("resolve", resolve), ("split", split), ("extract", extract), ("evaluate", evaluate)]:
        builder.add_node(name, fn)
    builder.add_edge(START, "resolve")
    builder.add_edge("resolve", "split")
    builder.add_edge("split", "extract")
    builder.add_conditional_edges("extract", lambda state: END if state.get("error") else "evaluate")
    if runtime is None:
        builder.add_edge("evaluate", END)
    else:
        builder.add_node("alternatives", alternatives)
        builder.add_node("review", review)
        builder.add_edge("evaluate", "alternatives")
        def route(state):
            labels = {o["result"]["label"] for o in state.get("options", {}).values()}
            return "review" if len(labels | {state["trace"].label}) > 1 else END
        builder.add_conditional_edges("alternatives", route)
        builder.add_edge("review", END)
    return builder.compile()


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        rows = list(reader)
        if not reader.fieldnames or "thread_id" not in reader.fieldnames:
            raise ValueError(f"Missing thread_id in {path}")
    ids = [r["thread_id"] for r in rows]
    if any(not i for i in ids) or len(set(ids)) != len(ids) or any(None in r or None in r.values() for r in rows):
        raise ValueError(f"Malformed or duplicate rows in {path}")
    return rows


def validate(path, ids):
    with path.open(newline="", encoding="utf-8") as stream:
        if next(csv.reader(stream), None) != ["thread_id", "label"]:
            raise ValueError("Invalid submission header")
    rows = read_rows(path)
    if [r["thread_id"] for r in rows] != ids:
        raise ValueError("Submission ID order differs from sample")
    pattern = DecisionTrace.model_fields["label"].metadata[0].pattern
    if any(not re.fullmatch(pattern, r["label"]) for r in rows):
        raise ValueError("Invalid submission label")
    return len(rows)


def export(path, rows):
    with NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        writer = csv.writer(stream)
        writer.writerow(["thread_id", "label"])
        writer.writerows(rows)
    try:
        ids = [i for i, _ in rows]
        validate(tmp, ids)
        tmp.replace(path)
        return validate(path, ids)
    finally:
        tmp.unlink(missing_ok=True)


def batch(rows, ids, book, policy_hash, directory, evaluation=False, runtime=None):
    indexed = {r["thread_id"]: r for r in rows}
    if set(ids) != indexed.keys() or len(ids) != len(indexed):
        raise ValueError("Input IDs differ from expected IDs")
    engine = digest([Path(p).read_text() for p in (baseline.__file__, __file__, Path(__file__).with_name("models.py"))])
    directory.mkdir(parents=True, exist_ok=True)
    flow = graph(book, policy_hash, runtime)
    config = "baseline"
    if runtime:
        from .providers import DEFAULTS
        env = runtime.registry.env
        config = [runtime.profile, runtime.registry.profile, sorted(runtime.qualified),
                  {k:v for k,v in env.items() if (k in DEFAULTS or k.endswith(("_CONCURRENCY","_TIMEOUT_SECONDS")))
                   and not k.endswith(("_KEY","_TOKEN"))},
                  [Path(__file__).with_name(f).read_text() for f in ("providers.py","integration.py","review.py")]]
    labels, errors, hits = {}, {}, 0
    with sqlite3.connect(directory/"state.sqlite3") as db:
        db.execute("CREATE TABLE IF NOT EXISTS decisions (key TEXT PRIMARY KEY, trace TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS snapshot (id INTEGER PRIMARY KEY, manifest TEXT NOT NULL)")
        manifest = []
        for i in ids:
            key = digest([indexed[i], policy_hash, engine, config])
            cached = db.execute("SELECT trace FROM decisions WHERE key=?", (key,)).fetchone()
            if cached:
                trace = DecisionTrace.model_validate_json(cached[0])
                hits += 1
            else:
                try:
                    state = flow.invoke({"thread": indexed[i]}, {"recursion_limit": 10})
                    if state.get("error"):
                        errors[i] = state["error"]
                        continue
                    trace = state["trace"]
                except (KeyError, ValueError, IndexError, ArithmeticError, StopIteration) as exc:
                    errors[i] = f"{type(exc).__name__}: {exc}"
                    continue
                db.execute("INSERT OR IGNORE INTO decisions VALUES (?,?)", (key, trace.model_dump_json()))
                db.commit()
            labels[i] = trace.label
            manifest.append([i, key])
        if not errors and not evaluation:
            db.execute("INSERT OR REPLACE INTO snapshot VALUES (1,?)", (json.dumps(manifest),))
            export(directory/"submission.csv", list(labels.items()))
    return labels, errors, hits


def replay(directory, output):
    with sqlite3.connect((directory/"state.sqlite3").resolve().as_uri()+"?mode=ro", uri=True) as db:
        snapshot = db.execute("SELECT manifest FROM snapshot WHERE id=1").fetchone()
        if not snapshot:
            raise ValueError("No complete batch snapshot")
        rows = []
        for i, key in json.loads(snapshot[0]):
            trace = DecisionTrace.model_validate_json(db.execute("SELECT trace FROM decisions WHERE key=?", (key,)).fetchone()[0])
            if trace.thread_id != i:
                raise ValueError("Trace ID differs from manifest")
            rows.append((i, trace.label))
    return export(output, rows)


def span(block, start, end):
    return Evidence(message_id=block.message_id, start=block.start + start, end=block.start + end,
                    excerpt=block.excerpt[start:end])


def clause_findings(messages):
    """Inspect security across all blocks and binding terms in the latest block."""
    security, contract = [], []
    for block in messages:
        for line in re.finditer(r"[^\n]+", block.excerpt):
            if re.search(r"AI सहायक|AI AGENT|whichever bot|language model|assistant:|automated system|previous instructions|pre-approved|override|reviewer,?\s+(?:approve|accept)|already (?:approved|granted)", line[0], re.I):
                security.append(ClauseFinding(kind="security", state="confirmed",
                                             evidence=(span(block, *line.span()),)))
    block = messages[-1]
    patterns = [
        ("negated", r"(?:need not|do not have to|are not required to) buy only from us|(?:does not|will not|never) auto-renew\w*|no automatic renewal|(?:not asking for|no) exclusivity"),
        ("confirmed", r"half the order value upfront|renews on its own|exclusive supplier|liability[^\n.]*unlimited|only from us|from year to year|auto-renews|automatic renewal"),
        ("advance", r"(\d+)% (?:advance|upfront|of the order value upfront)"),
    ]
    covered = []
    for state, pattern in patterns:
        for match in re.finditer(pattern, block.excerpt, re.I):
            if any(start <= match.start() < end for start, end in covered):
                continue
            if state == "advance" and int(match[1]) <= 30:
                continue
            contract.append(ClauseFinding(kind="contract", state="confirmed" if state == "advance" else state,
                                          evidence=(span(block, *match.span()),)))
            covered.append(match.span())
    return (security or [ClauseFinding(kind="security", state="absent")],
            contract or [ClauseFinding(kind="contract", state="absent")])


def decide(thread, effective_policy, normalized_offer, security_findings, contract_findings) -> DecisionTrace:
    """Pure projection of validated evidence through the baseline decision rules."""
    pol = effective_policy.fields
    result = baseline.decide(thread, pol, normalized_offer.unit_price,
                             any(f.state == "confirmed" for f in security_findings),
                             any(f.state == "confirmed" for f in contract_findings))
    operands = {name: baseline.number(pol[key]) for name, key in [
        ("target", "Target price"), ("walkaway", "Walk-away price"), ("R", "Negotiation rounds (R)"),
        ("beta", "Concession exponent (β)"), ("approval_limit", "Approval limit")]}
    operands.update(r=Decimal(thread["round"]), quantity=Decimal(thread["quantity"]))
    reason = result["label"].split(":")[1]
    lines = {"MISSING_INFO":"3-5", "BLACKLISTED_SUPPLIER":"6", "PROMPT_INJECTION":"7-8", "BANNED_TERM":"9-10",
             "ABOVE_WALKAWAY":"16", "COUNTER_OFFER":"17", "OVER_APPROVAL_LIMIT":"18-19", "ACCEPTABLE_PRICE":"19"}
    return DecisionTrace(**result, policy=effective_policy, offer=normalized_offer, operands=operands,
                         findings=tuple(security_findings)+tuple(contract_findings),
                         branch_source="policies/decision-procedure.md:"+lines[reason],
                         concession_source=("policies/decision-procedure.md:20" if "PROBATION" in pol["Status"]
                                            else "policies/concession-schedule.md:3"),
                         concession_formula="target" if "PROBATION" in pol["Status"] else
                         "target + (walkaway - target) * (r - 1)**beta / (R - 1)**beta")


def explain_offer(thread, policy, messages, choice=None):
    """Reconstruct the selected operands, checking the result against baseline.extract."""
    block = messages[-1]
    lines = [span(msg, *m.span()) for msg in messages for m in re.finditer(r"[^\n]+", msg.excerpt)]
    candidates = []
    for msg in messages:
        for m in baseline.MONEY.finditer(msg.excerpt):
            currency = {"€":"EUR", "£":"GBP", "$":"USD", "₹":"INR", "RS.":"INR"}.get(m[1].upper(), m[1].upper())
            candidates.append(QuoteCandidate(currency=currency, raw_amount=baseline.number(m[2]),
                                             evidence=span(msg, *m.span())))
    if choice:
        current = span(block, choice["start"], choice["end"])
        if current.excerpt != choice["excerpt"] or choice["message_id"] != block.message_id:
            raise ValueError("Selected span differs from supplier message")
        source_thread = dict(thread, supplier_message=current.excerpt)
    else:
        current = next((line for line in lines if line.message_id == block.message_id and
                        (baseline.MONEY.search(line.excerpt) or " rupees" in line.excerpt or
                         "off our previous quote" in line.excerpt)), None)
        source_thread = thread
    expected, _ = baseline.extract(source_thread, policy)
    if expected is None:
        return NormalizedOffer(unit_price=None, candidates=tuple(candidates), terms=(current,) if current else ())
    if current is None:
        raise ValueError("Priced offer lacks a source line")
    terms = [current]
    source = current
    if "off our previous quote" in current.excerpt:
        own = {m.message_id for m in messages[:-1] if m.speaker == "supplier" and
               thread["supplier_message"][:m.start].splitlines()[-1].lstrip("[").startswith(
                   "Earlier message — " + thread["supplier"])}
        source = next(line for line in reversed(lines) if line.message_id in own and baseline.MONEY.search(line.excerpt))
        terms.append(source)
    text = " ".join(t.excerpt for t in terms)
    quotes = [q for q in candidates if source.start <= q.evidence.start < source.end]
    tier = re.search(r"up to (\d+)", text)
    def scale(q):
        return 100000 if q.evidence.excerpt.lower().endswith("lakh") else 1000 if q.evidence.excerpt.lower().endswith("k") else 1
    if " rupees" in source.excerpt:
        words = "|".join([*baseline.WORDS, "hundred", "thousand", "lakh"])
        match = re.search(rf"\b(?:{words})(?:[ -](?:{words}))* rupees", source.excerpt)
        if not match:
            raise ValueError("Cannot identify written amount")
        selected = QuoteCandidate(currency="INR", raw_amount=baseline.word_price(source.excerpt),
                                  evidence=Evidence(message_id=source.message_id, start=source.start+match.start(),
                                                    end=source.start+match.end(), excerpt=match[0]))
        candidates.append(selected)
    elif "between " in text:
        selected = max(quotes, key=lambda q: q.raw_amount * Decimal(policy["FX "+q.currency]) * scale(q))
    else:
        selected = quotes[int(bool(tier and int(thread["quantity"]) > int(tier[1])))]
    basis = re.search(r"in total|per [\w-]+", source.excerpt)
    billing = re.search(r"per (month|year)", source.excerpt)
    selected = selected.model_copy(update={"price_basis": basis[0] if basis else None,
        "billing_period": billing[1] if billing else None,
        "quantity_interval": ((0, int(tier[1])) if selected == quotes[0] else (int(tier[1])+1, None)) if tier else None,
        "named_reference": "our previous quote" if len(terms) > 1 else None})
    value = selected.raw_amount
    steps = []
    units = "policies/units-and-currency.md:"
    def apply(rule, evidence, citation, numerator=1, denominator=1):
        nonlocal value
        before = value
        value = before * numerator / denominator
        steps.append(ConversionStep(rule=rule, source=citation, evidence=evidence, before=before,
                                    numerator=numerator, denominator=denominator, after=value))
    if selected.currency != "INR":
        key = "FX "+selected.currency
        apply("FX", selected.evidence, policy["source."+key], Decimal(policy[key]))
    if scale(selected) != 1:
        apply("notation", selected.evidence, units+"22", scale(selected))
    unit = policy["Normalized unit"]
    if "year" in unit and "year" not in text:
        apply("annualize", source, units+"16", 12)
    if "month" in unit and "month" not in text:
        apply("monthly", source, units+"16", denominator=12)
    discount = re.search(r"(\d+)% (?:off|discount)", text)
    threshold = re.search(r"off for orders of (\d+)", text)
    if discount and (not threshold or int(thread["quantity"]) >= int(threshold[1])):
        factor = 1 - Decimal(discount[1])/100
        apply("discount", current, units+("24" if len(terms)>1 else "20" if threshold else "18"), factor)
    free = re.search(r"for every (\d+) you pay", text)
    if free:
        apply("free unit", current, "precedents/P-02.md:3", int(free[1]), 1+int(free[1]))
    for pattern, name, citation in [(r"in total for all ([\d,]+)", "total", units+"17"),
                                    (r"per (\d+) \w+", "bundle", units+"21")]:
        match = re.search(pattern, text)
        if match:
            apply(name, source, citation, denominator=baseline.number(match[1]))
    if "per dozen" in text:
        apply("dozen", source, units+"21", denominator=12)
    if "per pack" in text:
        apply("pack", source, policy["source.Pack conversion"], denominator=baseline.number(policy["Pack conversion"]))
    if "per ream" in text:
        apply("ream", source, policy["source.Pack conversion"], baseline.number(policy["Pack conversion"]))
    if re.search(r"(?:incl(?:uding|usive of|\.)? GST|GST included)", text, re.I):
        apply("remove GST", source, policy["source.GST rate"], denominator=1+baseline.number(policy["GST rate"])/100)
    if value != expected:
        raise ValueError("Normalization operands disagree with baseline.extract")
    return NormalizedOffer(unit_price=value, candidates=tuple(candidates), selected=selected,
                           terms=tuple(terms), conversions=tuple(steps))
