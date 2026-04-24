"""Deterministic agreement-intro party and role resolution helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(slots=True, frozen=True)
class PartyRoleAssignment:
    parties: tuple[str, ...]
    employer: str | None
    employee: str | None
    company_side_party: str | None
    individual_side_party: str | None


_ROLE_ALIASES = {
    "employer": "employer",
    "employee": "employee",
    "company": "company_side_party",
    "company side": "company_side_party",
    "individual": "individual_side_party",
    "individual side": "individual_side_party",
}


def extract_intro_party_role_assignment(text: str) -> PartyRoleAssignment | None:
    raw = (text or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if not _has_intro_anchor(lowered):
        return None

    parties = list(_extract_between_or_parties_are(raw))
    role_values = _extract_role_values(raw)
    employer = role_values.get("employer")
    employee = role_values.get("employee")
    company_side_party = role_values.get("company_side_party")
    individual_side_party = role_values.get("individual_side_party")

    if not company_side_party:
        company_side_party = employer
    if not individual_side_party:
        individual_side_party = employee

    if len(parties) >= 2:
        first, second = parties[0], parties[1]
        first_role = _detect_inline_role(first)
        second_role = _detect_inline_role(second)
        if first_role == "employer":
            employer = employer or first
        elif first_role == "employee":
            employee = employee or first
        if second_role == "employer":
            employer = employer or second
        elif second_role == "employee":
            employee = employee or second

    if len(parties) >= 2 and (employer is None or employee is None):
        inferred_employer, inferred_employee = _infer_employer_employee(parties[0], parties[1], lowered)
        employer = employer or inferred_employer
        employee = employee or inferred_employee

    if employer and _is_placeholder_party(employer):
        employer = None
    if employee and _is_placeholder_party(employee):
        employee = None
    if company_side_party and _is_placeholder_party(company_side_party):
        company_side_party = None
    if individual_side_party and _is_placeholder_party(individual_side_party):
        individual_side_party = None
    parties = [party for party in parties if not _is_placeholder_party(party)]

    if employer and employee and normalize_party_text(employer) == normalize_party_text(employee):
        return None

    if not parties:
        from_roles = [value for value in (employer, employee, company_side_party, individual_side_party) if value]
        deduped = _dedupe_parties(from_roles)
        if len(deduped) >= 2:
            parties = deduped[:2]

    if len(parties) < 2:
        return None

    if company_side_party is None:
        company_side_party = employer or pick_company_party(parties)
    if individual_side_party is None:
        individual_side_party = employee or pick_individual_party(parties)

    return PartyRoleAssignment(
        parties=tuple(parties[:2]),
        employer=employer,
        employee=employee,
        company_side_party=company_side_party,
        individual_side_party=individual_side_party,
    )


def parse_party_verification_query_entities(lowered_query: str) -> dict[str, object] | None:
    between_match = re.search(r"\bis\s+(?:this|the)\s+agreement\s+between\s+(.+?)\s+and\s+(.+?)\??$", lowered_query)
    if between_match:
        first = normalize_party_text(between_match.group(1))
        second = normalize_party_text(between_match.group(2))
        ambiguous = not is_usable_party_entity(first) or not is_usable_party_entity(second) or first == second
        return {"targets": (first, second), "ambiguous": ambiguous}

    single_party_match = re.search(r"\bis\s+(?:this|the)\s+agreement\s+(?:with|for)\s+(.+?)\??$", lowered_query)
    if single_party_match:
        target = normalize_party_text(single_party_match.group(1))
        return {"targets": (target,), "ambiguous": not is_usable_party_entity(target)}
    return None


def compare_query_entities_against_extracted_parties(
    *,
    verification_targets: tuple[str, ...],
    extracted_parties: Sequence[str],
) -> dict[str, str]:
    normalized_extracted = [
        normalize_party_text(party)
        for party in extracted_parties
        if is_usable_party_entity(normalize_party_text(party))
    ]
    extracted_set = set(normalized_extracted)
    if len(extracted_set) < 2:
        return {"status": "incomplete_party_set"}

    target_set = set(verification_targets)
    if len(target_set) != len(verification_targets) or not target_set:
        return {"status": "query_ambiguous"}

    if all(target in extracted_set for target in verification_targets):
        return {"status": "matched"}
    return {"status": "mismatched"}


def pick_company_party(parties: Sequence[str]) -> str | None:
    for party in parties:
        if looks_like_organization(party):
            return party
    return None


def pick_individual_party(parties: Sequence[str]) -> str | None:
    for party in parties:
        if not looks_like_organization(party):
            return party
    return None


def looks_like_organization(value: str) -> bool:
    lowered = value.lower()
    org_tokens = (
        " inc",
        " llc",
        " ltd",
        " corp",
        " corporation",
        " company",
        " co.",
        " limited",
        " plc",
        " lp",
        " llp",
        " holdings",
        " group",
    )
    if any(token in f" {lowered}" for token in org_tokens):
        return True
    return bool(re.search(r"\b[a-z]+\s+(?:inc\.?|llc|ltd\.?|corp\.?|company|holdings|group)\b", lowered))


def normalize_party_text(value: str) -> str:
    lowered = (value or "").lower().strip()
    lowered = re.sub(r"[\"'“”]", "", lowered)
    lowered = re.sub(r"\([^)]*\)", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\b(the|this|that|an|a)\b", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip(" ,;:-")


def is_usable_party_entity(value: str) -> bool:
    normalized = normalize_party_text(value)
    if not normalized:
        return False
    if _is_placeholder_party(normalized):
        return False
    return len(normalized) >= 2


def _has_intro_anchor(lowered: str) -> bool:
    return bool(
        re.search(
            r"\b(this\s+.+?\s+agreement\s+is\s+made(?:\s+effective)?|by\s+and\s+between|between\s*:|between|parties\s+to\s+this\s+agreement\s+are|as\s+employer\s+and.+as\s+employee)\b",
            lowered,
        )
    )


def _extract_between_or_parties_are(text: str) -> tuple[str, ...]:
    patterns = (
        r"\bbetween\s*:?\s+(.+?)\s+\band\s*:?\s+(.+?)(?:[.;\n]|$)",
        r"\bparties\s+to\s+this\s+agreement\s+are\s+(.+?)\s+\band\s+(.+?)(?:[.;\n]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        first = _clean_party_name(match.group(1))
        second = _clean_party_name(match.group(2))
        values = tuple(value for value in (first, second) if value)
        if len(values) >= 2:
            return values[:2]
    return ()


def _extract_role_values(text: str) -> dict[str, str]:
    role_values: dict[str, str] = {}
    patterns = (
        re.compile(
            r"(?P<entity>[A-Za-z0-9][^;\n.]{1,140}?)\s*\(\s*(?:the\s+)?[\"'“”]?(?P<role>employer|employee|company(?:\s+side)?|individual(?:\s+side)?)"
            r"[\"'“”]?\s*\)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?P<entity>[A-Za-z0-9][^;\n.]{1,140}?)\s+as\s+(?:the\s+)?(?P<role>employer|employee|company(?:\s+side)?|individual(?:\s+side)?)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<role>employer|employee|company(?:\s+side)?|individual(?:\s+side)?)\s*[:\-]\s*(?P<entity>[^;\n.]+)",
            flags=re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            role_key = _ROLE_ALIASES.get(normalize_party_text(match.group("role")))
            entity = _clean_party_name(match.group("entity"))
            if role_key and entity and not _is_placeholder_party(entity):
                role_values.setdefault(role_key, entity)
    return role_values


def _clean_party_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\((?:the\s+)?[\"“']?(?:employer|employee|company(?:\s+side)?|individual(?:\s+side)?)[\"”']?\)", " ", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"[\"'“”]+", " ", cleaned)
    cleaned = re.sub(r"\b(?:the\s+)?(employer|employee|company|individual|party|parties)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned or None


def _detect_inline_role(value: str) -> str | None:
    lowered = value.lower()
    if "employer" in lowered or "company" in lowered:
        return "employer"
    if "employee" in lowered or "individual" in lowered:
        return "employee"
    return None


def _infer_employer_employee(first: str, second: str, lowered_text: str) -> tuple[str | None, str | None]:
    if "employment agreement" not in lowered_text:
        return None, None
    first_is_org = looks_like_organization(first)
    second_is_org = looks_like_organization(second)
    if first_is_org == second_is_org:
        return None, None
    if first_is_org:
        return first, second
    return second, first


def _is_placeholder_party(value: str) -> bool:
    normalized = normalize_party_text(value)
    placeholders = {
        "company",
        "the company",
        "employee",
        "the employee",
        "party",
        "the party",
        "parties",
        "the parties",
        "employer",
        "the employer",
        "party a",
        "party b",
        "first party",
        "second party",
    }
    return normalized in placeholders


def _dedupe_parties(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_party_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value)
    return deduped
