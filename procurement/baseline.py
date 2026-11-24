"""Offline evaluator for Acme's procurement playbook."""
import csv
import json
import re
from decimal import Decimal as D, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "the-procurement-desk/playbook/playbook"
MONEY = re.compile(r"(₹|Rs\.|INR|USD|EUR|GBP|\$|€|£)\s*([\d,]+(?:\.\d+)?)\s*(lakh|k\b)?", re.I)


def rows(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    return D(re.search(r"[\d,]+(?:\.\d+)?", value)[0].replace(",", ""))


def compile_policy(root=None):
    cats, vendors, changes = {}, {}, []
    root = Path(root or BOOK)
    fx_fields = {"FX INR": "1"}
    for index, line in enumerate((root / "policies/units-and-currency.md").read_text().splitlines(), 1):
        match = re.fullmatch(r"\| ([A-Z]{3}) \| ([\d.]+) \|", line)
        if match:
            key = f"FX {match[1]}"
            fx_fields[key] = match[2]
            fx_fields["source."+key] = f"policies/units-and-currency.md:{index}"
    for path in sorted(root.glob("*/*.md")):
        lines = path.read_text().splitlines()
        fields = dict(re.findall(r"\*\*(.+?):\*\* (.+)", "\n".join(lines)))
        sources = {f"source.{m[1]}":f"{path.relative_to(root)}:{i}"
                   for i, line in enumerate(lines, 1) if (m := re.search(r"\*\*(.+?):\*\*", line))}
        fields.update(sources)
        if path.parent.name == "categories":
            cats[path.stem] = fx_fields | fields | {"title": lines[0].split(": ")[1]}
        elif path.parent.name == "suppliers":
            vendors[lines[0].split(": ")[1]] = fields
        elif path.parent.name == "amendments":
            changes.append((fields["Effective from"], lines[0].split(": ")[1], fields["Change"], fields["source.Change"]))
    return cats, vendors, sorted(changes)


def resolve(thread, book):
    cats, vendors, changes = book
    cat = cats[thread["category"]].copy()
    vendor = vendors[thread["supplier"]]
    for date, title, change, source in changes:
        if title == cat["title"] and date <= thread["date"]:
            key = next(k for k in cat if k.lower() in change.lower() and k != "title")
            cat[key] = change.split("now ")[1]
            cat["source."+key] = source
    contract = vendor.get("Contract", "")
    if "price cap for " + cat["title"] + " of" in contract:
        cat["Walk-away price"] = contract
        cat["source.Walk-away price"] = vendor["source.Contract"]
    exchange = re.search(r"invoices in ([A-Z]{3}) are converted at ₹([\d.]+)", contract)
    if exchange:
        key = "FX " + exchange[1]
        cat[key] = exchange[2]
        cat["source."+key] = vendor["source.Contract"]
    return cat | vendor


WORDS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
WORDS.update(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))


def word_price(line):
    total = part = 0
    for word in line.split(" rupees")[0].replace("-", " ").split():
        if word in WORDS:
            part += WORDS[word]
        elif word == "hundred":
            part *= 100
        elif word in ("thousand", "lakh"):
            total += part * (1000 if word == "thousand" else 100000)
            part = 0
    return D(total + part)


def extract(thread, policy):
    message = thread["supplier_message"]
    latest = re.split(r"\[Latest message[^\]]*\]", message)[-1]
    quantity = int(thread["quantity"])
    line = next((s for s in latest.splitlines() if MONEY.search(s) or " rupees" in s or "off our previous quote" in s), "")
    if "previous quote" in line:
        history = re.findall(r"^\[Earlier message — ([^\]]+)\](.*?)(?=^\[(?:Earlier|Latest) message|\Z)", message, re.M | re.S)
        own = [body for header,body in history if header.split(",")[0] == thread["supplier"]]
        line += " " + next(s for body in reversed(own) for s in reversed(body.splitlines()) if MONEY.search(s))
    matches = list(MONEY.finditer(line))
    if " rupees" in line:
        price = word_price(line)
    elif matches:
        def in_rupees(match):
            cur, amount, scale = match.groups()
            iso = {"€": "EUR", "£": "GBP", "$": "USD", "₹": "INR", "RS.": "INR"}.get(cur.upper(), cur.upper())
            factor = {None: 1, "lakh": 100000, "k": 1000}[scale.lower() if scale else None]
            return number(amount) * D(policy["FX "+iso]) * factor

        prices = [in_rupees(m) for m in matches]
        tier = re.search(r"up to (\d+)", line)
        if "between " in line:
            price = max(prices)
        elif tier and quantity > int(tier[1]):
            price = prices[1]
        else:
            price = prices[0]
    elif not re.search(r"\d", latest) or re.search(r"final specs;|competitive price|\b(?:JPY|AED|SGD)\b", latest):
        return None, latest
    else:
        raise ValueError(latest)
    unit = policy["Normalized unit"]
    for period, other, factor in [("year","month",D(12)),("month","year",None)]:
        if period in unit and period not in line:
            if other not in line:
                return None, latest
            price = price * factor if factor else price / 12
    discount = re.search(r"(\d+)% (?:off|discount)", line)
    threshold = re.search(r"off for orders of (\d+)", line)
    if discount and (not threshold or quantity >= int(threshold[1])):
        price *= 1 - D(discount[1]) / 100
    free = re.search(r"for every (\d+) you pay", line)
    if free:
        price = price * D(free[1]) / (1 + D(free[1]))
    total = re.search(r"in total for all ([\d,]+)", line)
    if total:
        price /= number(total[1])
    bundle = re.search(r"per (\d+) \w+", line)
    if bundle:
        price /= D(bundle[1])
    if "per dozen" in line:
        price /= 12
    if "per pack" in line:
        price /= number(policy["Pack conversion"])
    if "per ream" in line:
        price *= number(policy["Pack conversion"])
    if re.search(r"(?:incl(?:uding|usive of|\.)? GST|GST included)", line, re.I):
        price /= 1 + number(policy["GST rate"]) / 100
    return price, latest


def decide(thread, effective_policy, normalized_offer, security_findings, contract_findings):
    pol = effective_policy
    p = normalized_offer
    r, rounds = D(thread["round"]), number(pol["Negotiation rounds (R)"])
    target, walk = number(pol["Target price"]), number(pol["Walk-away price"])
    limit = target
    if "PROBATION" not in pol["Status"]:
        exponent = number(pol["Concession exponent (β)"])
        limit += (walk-target)*(r-1)**exponent/(rounds-1)**exponent
    amount = p * D(thread["quantity"]) if p is not None else None
    if p is None:
        label = "ESCALATE:MISSING_INFO:"
    elif "BLACKLISTED" in pol["Status"]:
        label = "REJECT:BLACKLISTED_SUPPLIER:"
    elif security_findings:
        label = "ESCALATE:PROMPT_INJECTION:"
    elif contract_findings:
        label = "ESCALATE:BANNED_TERM:"
    elif p > limit and r == rounds:
        label = "REJECT:ABOVE_WALKAWAY:"
    elif p > limit:
        label = f"COUNTER:COUNTER_OFFER:{limit.quantize(D(1), rounding=ROUND_HALF_UP)}"
    elif amount > number(pol["Approval limit"]):
        label = "ESCALATE:OVER_APPROVAL_LIMIT:"
    else:
        label = "ACCEPT:ACCEPTABLE_PRICE:"
    return dict(thread_id=thread["thread_id"],label=label,price=p,limit=limit,approval_amount=amount)


def evaluate(thread, book):
    policy = resolve(thread, book)
    p, latest = extract(thread, policy)
    injection = re.search(r"AI सहायक|AI AGENT|whichever bot|language model|assistant:|automated system|previous instructions|pre-approved|override|reviewer,?\s+(?:approve|accept)|already (?:approved|granted)", thread["supplier_message"], re.I)
    binding = re.sub(r"(?:need not|do not have to|are not required to) buy only from us|(?:does not|will not|never) auto-renew|no automatic renewal", "", latest, flags=re.I)
    banned = re.search(r"half the order value upfront|renews on its own|exclusive supplier|liability.*unlimited|only from us|from year to year|auto-renews|automatic renewal", binding, re.I)
    advances = re.findall(r"(\d+)% (?:advance|upfront|of the order value upfront)", latest)
    return decide(thread, policy, p, injection, banned or any(int(a)>30 for a in advances))


def main():
    book = compile_policy()
    output = ROOT / "runs/best"
    answers = {r["thread_id"]:r["label"] for r in rows("sample_answers.csv")}
    for name in ["sample_threads.csv","threads.csv"]:
        predictions, errors = {}, []
        for thread in rows(name):
            try:
                trace = evaluate(thread, book)
                predictions[thread["thread_id"]] = trace
            except (ValueError, IndexError, KeyError, StopIteration) as exc:
                errors.append((thread["thread_id"],str(exc)))
        print(name,len(predictions),len(errors), errors[:3])
        if name.startswith("sample"):
            misses = [(i, answers[i], predictions.get(i,{}).get("label")) for i in answers if answers[i]!=predictions.get(i,{}).get("label")]
            report = dict(matches=len(answers)-len(misses),total=len(answers),misses=misses)
            print(report)
            (output/"evaluation.json").write_text(json.dumps(report,indent=2))
        elif not errors:
            ids = [r["thread_id"] for r in rows("sample_submission.csv")]
            assert len(ids)==len(set(ids))==len(predictions) and set(ids)==predictions.keys()
            expected = [dict(thread_id=i,label=predictions[i]["label"]) for i in ids]
            tmp = output/"submission.tmp"
            with tmp.open("w",newline="") as f:
                writer = csv.DictWriter(f,fieldnames=["thread_id","label"])
                writer.writeheader()
                writer.writerows(expected)
            assert rows(tmp)==expected
            tmp.replace(output/"submission.csv")
            (output/"traces.jsonl").write_text("".join(json.dumps(t,default=str)+"\n" for t in predictions.values()))

if __name__ == "__main__":
    main()
