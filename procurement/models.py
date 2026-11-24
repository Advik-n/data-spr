"""Validated records shared by the workflow and its replay store."""
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Evidence(Record):
    message_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    excerpt: str

    def verify(self, text: str):
        if self.end > len(text) or self.start >= self.end or text[self.start:self.end] != self.excerpt:
            raise ValueError("Evidence offsets differ from source text")


class Message(Evidence):
    speaker: Literal["buyer", "supplier"]


class QuoteCandidate(Record):
    raw_amount: Decimal
    currency: str
    price_basis: str | None = None
    billing_period: str | None = None
    quantity_interval: tuple[int, int | None] | None = None
    named_reference: str | None = None
    evidence: Evidence


class ClauseFinding(Record):
    kind: Literal["contract", "security"]
    state: Literal["confirmed", "negated", "conditional", "unresolved", "absent"]
    evidence: tuple[Evidence, ...] = ()


class EffectivePolicy(Record):
    fields: dict[str, str]
    policy_hash: str


class ConversionStep(Record):
    rule: str
    source: str
    evidence: Evidence
    before: Decimal
    numerator: Decimal
    denominator: Decimal
    after: Decimal


class NormalizedOffer(Record):
    unit_price: Decimal | None
    candidates: tuple[QuoteCandidate, ...]
    selected: QuoteCandidate | None = None
    terms: tuple[Evidence, ...] = ()
    conversions: tuple[ConversionStep, ...] = ()


class DecisionTrace(Record):
    thread_id: str
    label: str = Field(pattern=r"^(?:ACCEPT:ACCEPTABLE_PRICE:|REJECT:(?:BLACKLISTED_SUPPLIER|ABOVE_WALKAWAY):|ESCALATE:(?:MISSING_INFO|PROMPT_INJECTION|BANNED_TERM|OVER_APPROVAL_LIMIT):|COUNTER:COUNTER_OFFER:[0-9]+)$")
    price: Decimal | None
    limit: Decimal
    approval_amount: Decimal | None
    policy: EffectivePolicy
    offer: NormalizedOffer
    review: dict = Field(default_factory=dict)
    messages: tuple[Message, ...] = ()
    findings: tuple[ClauseFinding, ...] = ()
    concession_formula: str = ""
    concession_source: str = ""
    approval_formula: str = "unit_price * quantity"
    operands: dict[str, Decimal] = Field(default_factory=dict)
    branch_source: str = ""
