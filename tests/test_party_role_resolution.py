from __future__ import annotations

from agentic_rag.tools.party_role_resolution import (
    compare_query_entities_against_extracted_parties,
    extract_intro_party_role_assignment,
    parse_party_verification_query_entities,
)


def test_role_value_normalization_strips_trailing_location_and_preserves_names() -> None:
    text = (
        "This Employment Agreement is made effective January 1, 2025, by and between "
        "Aurora Data Systems Inc., Toronto, Ontario (the “Employer”) and "
        "Daniel Reza Mohammadi, Toronto, Ontario (the “Employee”)."
    )

    assignment = extract_intro_party_role_assignment(text)

    assert assignment is not None
    assert assignment.employer == "Aurora Data Systems Inc."
    assert assignment.employee == "Daniel Reza Mohammadi"
    assert assignment.parties == ("Aurora Data Systems Inc.", "Daniel Reza Mohammadi")
    assert assignment.company_side_party == "Aurora Data Systems Inc."
    assert assignment.individual_side_party == "Daniel Reza Mohammadi"


def test_company_legal_suffix_with_comma_is_preserved_when_location_is_trimmed() -> None:
    text = (
        "This Employment Agreement is made by and between "
        "Acme Corp, Inc., Seattle, Washington (the “Employer”) and "
        "Jane Smith, Seattle, Washington (the “Employee”)."
    )

    assignment = extract_intro_party_role_assignment(text)

    assert assignment is not None
    assert assignment.employer == "Acme Corp, Inc."
    assert assignment.employee == "Jane Smith"
    assert assignment.parties == ("Acme Corp, Inc.", "Jane Smith")


def test_agreement_between_validation_uses_normalized_party_names_after_location_trimming() -> None:
    text = (
        "This Employment Agreement is made by and between "
        "Aurora Data Systems Inc., Toronto, Ontario (the “Employer”) and "
        "Daniel Reza Mohammadi, Toronto, Ontario (the “Employee”)."
    )
    assignment = extract_intro_party_role_assignment(text)

    assert assignment is not None
    parsed_query = parse_party_verification_query_entities(
        "is this agreement between aurora data systems inc and daniel reza mohammadi?"
    )
    assert parsed_query is not None
    assert parsed_query["ambiguous"] is False

    verification = compare_query_entities_against_extracted_parties(
        verification_targets=parsed_query["targets"],
        extracted_parties=assignment.parties,
    )
    assert verification["status"] == "matched"
