"""Offline inventory: python -m procurement.providers list
Local probes: python -m procurement.providers probe
Hosted probes: python -m procurement.providers probe --profile hosted --provider groq
Share one Registry per run and one row ID across direct and delegated calls.
"""
import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from threading import Lock, BoundedSemaphore
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

DEFAULTS = {
    "LLM_FREE_ONLY": "true",
    "LLM_PROVIDER_ORDER": "ollama,openrouter,groq,nvidia,gemini,cerebras,huggingface,cloudflare,custom",
    "LLM_CONFIRMED_NO_CHARGE_PROVIDERS": "",
    "LLM_TIMEOUT_SECONDS": "45",
    "LLM_MAX_CALLS_PER_ROW": "3",
    "LLM_MAX_CALLS_PER_RUN": "240",
    "LLM_MAX_POLICY_AUDIT_CALLS": "16",
    "OPENROUTER_API_KEY": "",
    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
    "OPENROUTER_MODELS": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free,inclusionai/ling-3.2-flash:free,recursive/flint-v1.0:free",
    "OPENROUTER_DISCOVER_FREE": "true",
    "OPENROUTER_FREE_ROUTER_MODEL": "openrouter/free",
    "OPENROUTER_USE_FREE_ROUTER": "true",
    "GROQ_API_KEY": "",
    "GROQ_BASE_URL": "https://api.groq.com/openai/v1",
    "GROQ_MODELS": "openai/gpt-oss-120b,openai/gpt-oss-20b",
    "NVIDIA_API_KEY": "",
    "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
    "NVIDIA_MODELS": "nvidia/nemotron-3-super-120b-a12b",
    "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
    "OLLAMA_MODELS": "qwen2.5-coder:7b-instruct,qwen3.5:0.8b",
    "GEMINI_API_KEY": "",
    "GEMINI_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "GEMINI_MODELS": "gemini-3.5-flash-lite,gemini-2.5-flash-lite",
    "CEREBRAS_API_KEY": "",
    "CEREBRAS_BASE_URL": "https://api.cerebras.ai/v1",
    "CEREBRAS_MODELS": "gpt-oss-120b",
    "HF_TOKEN": "",
    "HF_BASE_URL": "https://router.huggingface.co/v1",
    "HF_MODELS": "reflection/Beam-1.0-501B-A23B:baseten",
    "CLOUDFLARE_API_TOKEN": "",
    "CLOUDFLARE_ACCOUNT_ID": "",
    "CLOUDFLARE_BASE_URL": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
    "CLOUDFLARE_MODELS": "@cf/openai/gpt-oss-20b",
    "CUSTOM_BASE_URL": "",
    "CUSTOM_API_KEY": "",
    "CUSTOM_MODELS": ""
}
PREFIX = {"huggingface": "HF"}
KEYS = {"huggingface": "HF_TOKEN", "cloudflare": "CLOUDFLARE_API_TOKEN"}


def settings(path):
    if not path.exists():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text().splitlines()
             if "=" in line and not line.lstrip().startswith("#"))
    return {k.strip(): v.strip().strip("\"'") for k, v in pairs}


def names(value):
    return [s.strip() for s in value.split(",") if s.strip()]


def flag(value):
    if value.lower() not in {"true", "false"}:
        raise ValueError("expected true or false")
    return value.lower() == "true"


@dataclass(frozen=True)
class Provider:
    alias: str
    url: str = field(repr=False)
    key: str = field(repr=False)
    models: tuple
    timeout: float
    concurrency: int
    status: str


def free_chat(model, required=()):
    price = model.get("pricing") or {}
    arch = model.get("architecture") or {}
    try:
        return ({"prompt", "completion"} <= price.keys()
                and all(Decimal(str(value)) == 0 for value in price.values())
                and "text" in arch.get("input_modalities", [])
                and "text" in arch.get("output_modalities", [])
                and "/api/v1/chat/completions" in model.get("endpoints", ["/api/v1/chat/completions"])
                and (model.get("expiration_date") or "9999") > date.today().isoformat()
                and set(required) <= set(model.get("supported_parameters", [])))
    except (InvalidOperation, TypeError):
        return False


def classify(exc):
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    detail = str(exc).lower()
    if status in (401, 403):
        return "authentication"
    if status == 402 or any(x in detail for x in ("insufficient_quota", "quota exhausted", "out of credits")):
        return "exhausted_quota"
    if status == 429:
        return "rate_limit"
    if status == 404 or "model_not_found" in detail:
        return "invalid_model"
    if status in (400, 422) or isinstance(exc, NotImplementedError):
        return "unsupported_feature"
    if isinstance(exc, (ValueError, KeyError, IndexError)):
        return "malformed_output"
    if isinstance(exc, ImportError):
        return "missing_dependency"
    return "transient"


def delay(exc):
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    for key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset"):
        value = str(headers.get(key, ""))
        try:
            number = float(value)
            return max(1, number - time.time() if number > 1e9 else number)
        except ValueError:
            if re.fullmatch(r"(?:\d+(?:\.\d+)?[ms])+", value):
                return sum(float(n) * (60 if u == "m" else 1)
                           for n, u in re.findall(r"(\d+(?:\.\d+)?)([ms])", value))
            try:
                return max(1, parsedate_to_datetime(value).timestamp() - time.time())
            except (ValueError, TypeError):
                pass
    return 30


class OperationalError(RuntimeError):
    pass


class Registry:
    def __init__(self, profile="baseline", env=None):
        if profile not in {"baseline", "local", "hosted"}:
            raise ValueError("unknown provider profile")
        self.env = DEFAULTS | (settings(Path(".env")) | dict(os.environ) if env is None else env)
        e = self.env
        self.profile = profile
        self.free = flag(e["LLM_FREE_ONLY"])
        self.providers = {}
        self.slots = {}
        self.cooldown = {}
        self.calls = Counter()
        self.total = self.audit = 0
        self.lock = Lock()
        confirmed = names(e["LLM_CONFIRMED_NO_CHARGE_PROVIDERS"])
        for alias in names(e["LLM_PROVIDER_ORDER"]):
            prefix = PREFIX.get(alias, alias.upper())
            url = e.get(prefix + "_BASE_URL", "").rstrip("/")
            account = e.get("CLOUDFLARE_ACCOUNT_ID", "")
            url = url.replace("{account_id}", account)
            key = e.get(KEYS.get(alias, prefix + "_API_KEY"), "")
            local = alias == "ollama" and urlsplit(url).hostname in {"127.0.0.1", "::1", "localhost"}
            official = alias == "openrouter" and url == "https://openrouter.ai/api/v1"
            status = "eligible"
            if profile == "baseline" or (profile == "local" and not local):
                status = "profile_disabled"
            elif not url or (not key and alias not in {"ollama", "custom"}) or (alias == "cloudflare" and not account):
                status = "unconfigured"
            elif self.free and not local and not official and alias not in confirmed:
                status = "unconfirmed_billing"
            concurrency = int(e.get(prefix + "_CONCURRENCY", 1))
            if concurrency < 1:
                raise ValueError("concurrency must be positive")
            self.providers[alias] = Provider(alias, url, key, tuple(names(e.get(prefix + "_MODELS", ""))),
                float(e.get(prefix + "_TIMEOUT_SECONDS", e["LLM_TIMEOUT_SECONDS"])), concurrency, status)
            self.slots[alias] = BoundedSemaphore(concurrency)

    def catalog(self, p, path):
        with urlopen(p.url + path, timeout=p.timeout) as response:
            return json.load(response)

    def models(self, p, required=()):
        if p.status != "eligible":
            return []
        if p.alias == "ollama":
            installed = [m["name"] for m in self.catalog(p, "/api/tags")["models"]]
            return [m for m in (p.models or installed) if m in installed and
                    (not self.free or not m.endswith((":cloud", "-cloud")))]
        if p.alias != "openrouter":
            return list(p.models)
        data = self.catalog(p, "/models")["data"]
        eligible = [m["id"] for m in data if free_chat(m, required)]
        ids = set(eligible if self.free else [m["id"] for m in data])
        router = self.env["OPENROUTER_FREE_ROUTER_MODEL"]
        choices = [m for m in p.models if m in ids and m != router]
        if flag(self.env["OPENROUTER_DISCOVER_FREE"]):
            choices.extend(sorted(m for m in eligible if m.endswith(":free")))
        if flag(self.env["OPENROUTER_USE_FREE_ROUTER"]) and router in ids:
            choices.append(router)
        return list(dict.fromkeys(choices))

    def reserve(self, row, audit):
        with self.lock:
            e = self.env
            if (self.total >= int(e["LLM_MAX_CALLS_PER_RUN"])
                or not audit and self.calls[row] >= int(e["LLM_MAX_CALLS_PER_ROW"])
                or audit and self.audit >= int(e["LLM_MAX_POLICY_AUDIT_CALLS"])):
                raise OperationalError("budget_exhausted")
            self.calls[row] += 1
            self.total += 1
            self.audit += int(audit)

    def client(self, p, model):
        if p.alias == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(base_url=p.url, model=model, temperature=0,
                              client_kwargs={"timeout": p.timeout})
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(base_url=p.url, api_key=p.key or "local", model=model,
                          timeout=p.timeout, max_retries=0, temperature=0)

    def _attempt(self, p, model, messages, schema, method, row, audit):
        with self.slots[p.alias]:
            if self.cooldown.get(p.alias, 0) > time.monotonic():
                raise OperationalError("cooldown")
            self.reserve(row, audit)
            started = time.monotonic()
            try:
                client = self.client(p, model)
                if method == "schema":
                    reply = client.with_structured_output(schema, method="json_schema", include_raw=True).invoke(messages)
                    raw, value = reply["raw"], reply["parsed"]
                    if value is None:
                        raise ValueError("missing structured response")
                elif method == "tool":
                    raw = client.bind_tools([schema]).invoke(messages)
                    calls = raw.tool_calls
                    if len(calls) != 1 or calls[0]["name"] != schema.__name__:
                        raise ValueError("unexpected tool call")
                    value = schema.model_validate(calls[0]["args"])
                elif method == "json":
                    instruction = "Return one JSON object matching " + json.dumps(schema.model_json_schema())
                    raw = client.invoke([("system", instruction), *messages])
                    value = schema.model_validate_json(raw.content)
                else:
                    raw = client.invoke(messages)
                    value = raw.content
                meta = raw.response_metadata
                return value, {"alias": p.alias, "model": model, "serving_model": meta.get("model_name") or meta.get("model"),
                               "method": method, "seconds": time.monotonic() - started}
            except Exception as exc:
                reason = classify(exc)
                if reason in {"authentication", "exhausted_quota"}:
                    self.cooldown[p.alias] = float("inf")
                elif reason in {"rate_limit", "transient"}:
                    self.cooldown[p.alias] = time.monotonic() + delay(exc)
                raise OperationalError(reason) from None

    def extract(self, messages, schema, row, audit=False):
        failures = []
        for p in self.providers.values():
            try:
                models = self.models(p)
            except Exception as exc:
                failures.append([p.alias, "catalog", classify(exc)])
                continue
            for model in models:
                for method in ("schema", "tool", "json"):
                    try:
                        value, meta = self._attempt(p, model, messages, schema, method, row, audit)
                        return value, dict(meta, failures=failures, calls=self.calls[row])
                    except OperationalError as exc:
                        failures.append([p.alias, model, method, str(exc)])
                        if str(exc) == "budget_exhausted":
                            raise OperationalError(json.dumps(failures)) from None
                        if str(exc) not in {"unsupported_feature", "malformed_output"}:
                            break
        raise OperationalError(json.dumps(failures) if failures else "no_eligible_route")

    def probe(self, aliases, cache, refresh=False):
        from pydantic import BaseModel
        class Quote(BaseModel):
            amount: int
        reports = []
        with sqlite3.connect(cache) as db:
            db.execute("CREATE TABLE IF NOT EXISTS probes (id TEXT PRIMARY KEY, at REAL, report TEXT)")
            for alias in aliases:
                p = self.providers[alias]
                try:
                    shortlist = self.models(p)[:3]
                except Exception as exc:
                    reports.append({"alias": alias, "status": classify(exc)})
                    continue
                if not shortlist:
                    reports.append({"alias": alias, "status": p.status if p.status != "eligible" else "no_models"})
                for model in shortlist:
                    key = hashlib.sha256(repr(("probe-v1", p.url, p.key, alias, model, p.timeout)).encode()).hexdigest()
                    hit = db.execute("SELECT report FROM probes WHERE id=? AND at>?", (key, time.time()-86400)).fetchone()
                    if hit and not refresh:
                        reports.append(json.loads(hit[0]))
                        continue
                    results = {}
                    for method in ("chat", "schema", "tool"):
                        prompt = {"chat": "Reply with exactly 17.", "schema": "Extract the unit price from: 17 INR per pen.",
                                  "tool": "Call Quote with the unit price from: 17 INR per pen."}[method]
                        try:
                            value, meta = self._attempt(p, model, [("user", prompt)], Quote, method, "probe/"+key, False)
                            ok = str(value).strip() == "17" if method == "chat" else value.amount == 17
                            results[method] = dict(meta, passed=ok)
                        except OperationalError as exc:
                            results[method] = {"passed": False, "status": str(exc)}
                    report = {"alias": alias, "model": model, "checks": results}
                    reports.append(report)
                    db.execute("INSERT OR REPLACE INTO probes VALUES (?,?,?)", (key, time.time(), json.dumps(report)))
                    db.commit()
        def rank(report):
            checks = report.get("checks", {}).values()
            return (-sum(c["passed"] for c in checks), sum(c.get("seconds", 0) for c in checks))
        return sorted(reports, key=rank)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "probe"])
    parser.add_argument("--profile", choices=["baseline", "local", "hosted"], default="local")
    parser.add_argument("--provider", action="append")
    parser.add_argument("--cache", default="provider-probes.sqlite3")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    registry = Registry(args.profile)
    aliases = args.provider or list(registry.providers)
    if set(aliases) - registry.providers.keys():
        parser.error("unknown provider alias")
    if args.command == "list":
        result = [{"alias": a, "status": registry.providers[a].status, "models": registry.providers[a].models}
                  for a in aliases]
    else:
        result = registry.probe(aliases, args.cache, args.refresh)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
