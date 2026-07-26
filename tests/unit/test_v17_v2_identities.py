from __future__ import annotations

import pytest

from quant_investor.v17_v2_contract.identities import (
    IdentityContractError,
    ascii_casefold_key,
    require_ascii_casefold_unique,
    require_opaque_id,
    require_path_id,
    require_registry_token,
    require_security_code,
    require_sha256,
)


def test_opaque_id_is_lowercase_ascii_and_is_never_normalized() -> None:
    assert require_opaque_id("run-20260723.cn:v2") == "run-20260723.cn:v2"
    for value in (
        "Run-20260723",
        " run-20260723",
        "run-20260723 ",
        "Ｒun",
        "é",
        "",
        "a" * 129,
        17,
    ):
        with pytest.raises(IdentityContractError):
            require_opaque_id(value)


def test_path_id_excludes_colon_and_path_syntax() -> None:
    assert require_path_id("run-20260723.cn_v2") == "run-20260723.cn_v2"
    for value in ("run:v2", "run/v2", "../run", "Run", "run\\v2", ""):
        with pytest.raises(IdentityContractError):
            require_path_id(value)


@pytest.mark.parametrize("value", ["000001.SZ", "600000.SH", "830001.BJ"])
def test_security_code_accepts_only_fully_qualified_canonical_codes(value: str) -> None:
    assert require_security_code(value) == value


@pytest.mark.parametrize(
    "value",
    ["000001", "000001.sz", "000001.SS", "1000000.SH", " 000001.SZ", 1],
)
def test_security_code_rejects_noncanonical_forms(value: object) -> None:
    with pytest.raises(IdentityContractError):
        require_security_code(value)


def test_sha256_is_lowercase_exact_hex() -> None:
    digest = "a" * 64
    assert require_sha256(digest) == digest
    for value in ("A" * 64, "a" * 63, "g" * 64, 0):
        with pytest.raises(IdentityContractError):
            require_sha256(value)


def test_registry_is_exact_match_and_ascii_casefold_unique() -> None:
    registry = ("READ_STATUS", "READ_ARTIFACT")
    assert require_ascii_casefold_unique(registry) == registry
    assert ascii_casefold_key("READ_STATUS") == "read_status"
    assert (
        require_registry_token(
            "READ_STATUS",
            registry=frozenset(registry),
            label="action",
        )
        == "READ_STATUS"
    )
    with pytest.raises(IdentityContractError, match="exact"):
        require_registry_token(
            "read_status",
            registry=frozenset(registry),
            label="action",
        )
    with pytest.raises(IdentityContractError, match="casefold collision"):
        require_ascii_casefold_unique(("Read", "READ"))
    with pytest.raises(IdentityContractError):
        require_ascii_casefold_unique(("READ", "ＲＥＡＤ"))
    with pytest.raises(IdentityContractError):
        require_registry_token("READ", registry=("READ", 1))  # type: ignore[arg-type]
