"""Differential old/new control predicates through the actual canonical validator."""

import pytest
from quant_investor.contracts import core


class _LegacyControlPattern:
    @staticmethod
    def search(value):
        # Exact predicate from the original implementation; test oracle only.
        return True if any(ord(character) < 0x20 for character in value) else None


def outcome(value):
    try:
        return "accepted", core.canonical_json_bytes(value)
    except Exception as exc:
        return "rejected", type(exc), str(exc)


def test_every_unicode_codepoint_matches_previous_predicate():
    for codepoint in range(0x110000):
        assert (core._CONTROL_CHARACTER_RE.search(chr(codepoint)) is not None) == (codepoint < 32)


@pytest.mark.parametrize("codepoint", range(32))
@pytest.mark.parametrize("position", [0, 2, 4])
def test_controls_in_values_and_keys_preserve_exact_errors(monkeypatch, codepoint, position):
    text = "abcd"[:position] + chr(codepoint) + "abcd"[position:]
    for value in [text, {text: 1}, {"nested": ["valid", {"value": text}]}]:
        actual = outcome(value)
        with monkeypatch.context() as patch:
            patch.setattr(core, "_CONTROL_CHARACTER_RE", _LegacyControlPattern)
            expected = outcome(value)
        assert actual == expected
        assert actual[0] == "rejected"


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        1,
        -1,
        0.0,
        -0.0,
        1.25,
        "",
        "plain ASCII",
        "中文😀é",
        "e\u0301",
        "\ud800",
        "\udfff",
        "\x7f\u0080\u009f",
        {"": 1},
        {"中文": 1},
        {1: "invalid key"},
        {"b": [False, None, {"a": "é"}], "a": 0},
        float("nan"),
        float("inf"),
        {"x": object()},
    ],
)
def test_canonical_bytes_and_rejection_order_unchanged(monkeypatch, value):
    actual = outcome(value)
    with monkeypatch.context() as patch:
        patch.setattr(core, "_CONTROL_CHARACTER_RE", _LegacyControlPattern)
        expected = outcome(value)
    assert actual == expected


def test_bounds_cycles_and_catalog_unchanged(monkeypatch):
    cycle = []
    cycle.append(cycle)
    nested = 0
    for _ in range(core.MAX_CANONICAL_DEPTH + 2):
        nested = [nested]
    # Real byte bounds, including a multibyte UTF-8 overrun before control scan.
    values = [
        cycle,
        nested,
        "x" * (core.MAX_CANONICAL_JSON_BYTES + 1),
        "é" * (core.MAX_CANONICAL_JSON_BYTES // 2 + 1),
    ]
    actual = [outcome(value) for value in values]
    catalog = core.contract_catalog_sha256()
    with monkeypatch.context() as patch:
        patch.setattr(core, "_CONTROL_CHARACTER_RE", _LegacyControlPattern)
        assert [outcome(value) for value in values] == actual
        assert core.contract_catalog_sha256() == catalog
    assert all(row[0] == "rejected" for row in actual)
