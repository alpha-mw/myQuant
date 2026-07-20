from __future__ import annotations

import base64
import copy
import csv
import hashlib
import importlib.metadata
import math
import sys
from pathlib import Path
from typing import TypedDict

import pytest

from quant_investor.factors import governance_prior_diagnostic_nomination_v4_3 as contract


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _PeriodData(TypedDict):
    counts: list[int]
    common: list[int]
    ics: list[float]


_PERIOD_DATA: dict[str, _PeriodData] = {
    "VOL_OF_VOL_20D": {
        "counts": [5049, 4972, 5032, 5079, 5138, 5165, 5178, 5189, 5199, 5234, 5253, 5258, 5223, 5218, 5256, 5275, 5288, 5283, 5280, 5277, 5296, 5305, 5327, 5315, 5229, 5239, 5330, 5347, 5364, 5374, 5373, 5382, 5374, 5397, 5419, 5435, 5315, 5344],
        "common": [4928, 4969, 5031, 5076, 5136, 5159, 5176, 5184, 5198, 5232, 5252, 5258, 5221, 5218, 5253, 5267, 5280, 5266, 5265, 5265, 5290, 5296, 5315, 5306, 5220, 5223, 5323, 5340, 5357, 5363, 5358, 5359, 5361, 5391, 5410, 5414, 5301, 5334],
        "ics": [0.08490633683444958, -0.1184149145901251, -0.35724968816590646, 0.018680762754034113, -0.14627115543724228, 0.06346387963916843, -0.037742059659621005, -0.20193665138008224, -0.22686263987429506, -0.12456209190704187, -0.23215674719700508, -0.3723262133420399, -0.19857284349172274, -0.14144727715109381, -0.05400475896301311, -0.052664174846772265, 0.2699287418998079, -0.06881628634147191, 0.11093030077925585, -0.37306728341783335, 0.1871344501541689, 0.23840285593940203, -0.2800006935736872, 0.030587252831265347, 0.06139535445177674, -0.13693312059841414, -0.10073103931965575, -0.019932711487300185, -0.22499022329086685, -0.2764578739277729, -0.0629689272556044, 0.0479178151928914, -0.060530764425551525, -0.18513750971130868, -0.1564265641747838, 0.038101489139639164, -0.03977093319169427, -0.08758809701112594],
    },
    "MOM_12M_SKIP1M": {
        "counts": [4990, 5008, 5040, 5078, 5129, 5161, 5186, 5191, 5209, 5226, 5245, 5261, 5266, 5268, 5272, 5279, 5288, 5296, 5299, 5310, 5308, 5324, 5334, 5345, 5337, 5355],
        "common": [4986, 5006, 5037, 5068, 5118, 5137, 5160, 5170, 5194, 5210, 5227, 5244, 5252, 5237, 5256, 5266, 5273, 5275, 5278, 5275, 5289, 5308, 5315, 5314, 5307, 5328],
        "ics": [0.05535913820233916, 0.1899180950565281, -0.20444464988706598, -0.1371222124940996, -0.13303304289628207, 0.03394250832746351, -0.2409054594960688, 0.018791003826108734, -0.03985197415177841, -0.07043890668650253, -0.06516719581234376, 0.02426040786009606, -0.1460299848935694, -0.08386485067667145, -0.10486735074902635, 0.01886029602419882, -0.25809087654339125, -0.3082171069021597, -0.13181434530064218, 0.174539324329313, -0.00309593232562054, -0.06559752735798331, -0.08041348167667964, 0.19573643319877412, 0.043393917419021544, -0.011786059942342681],
    },
    "EXCESS_MOM_60D": {
        "counts": [4841, 4917, 4952, 5011, 5012, 5033, 5040, 5054, 5078, 5088, 5083, 5067, 5068, 5087, 5095, 5092, 4996, 4975, 4963, 4962, 4972, 4976, 4985, 4974, 4987, 5071, 5082, 5109, 5021, 5025],
        "common": [4840, 4915, 4951, 5011, 5011, 5033, 5037, 5046, 5070, 5076, 5072, 5056, 5062, 5079, 5083, 5084, 4988, 4960, 4957, 4955, 4967, 4969, 4970, 4956, 4976, 5066, 5073, 5091, 5010, 5016],
        "ics": [-0.17766522922677128, -0.31165235303303623, 0.15834882779244583, 0.15884936877526254, 0.23083466745470252, 0.018305088154153052, -0.16478860222175526, -0.025238713851391543, 0.15679640678595924, 0.08241791764428522, 0.06108196610865755, -0.43278793247045827, -0.2716296074639032, -0.1638254921462473, -0.036266400771973586, -0.036109215020069446, 0.03649602894579606, -0.10868446595943033, -0.007074887677213497, -0.07706164797027222, -0.08604272985160745, -0.29137045598458283, 0.05045267792975035, -0.1786477455377176, -0.033949076429865103, -0.10530951822478667, -0.12832390674315966, -0.11099237661989651, 0.07775281426850825, 0.13043222469401758],
    },
}


def _reseal(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value.pop("artifact_semantic_sha256", None)
    value["artifact_semantic_sha256"] = contract.semantic_sha256_v4_3(value)
    return value


def _attempt(source_name: str) -> dict:
    expected = next(row for row in contract.ATTEMPT_SPECS if row["source_name"] == source_name)
    suffix = contract.EXPECTED_MONTHLY_DATES[
        contract.EXPECTED_MONTHLY_DATES.index(expected["effective_start"]):
    ]
    observed = _PERIOD_DATA[source_name]
    assert len(suffix) == len(observed["counts"]) == len(observed["common"]) == len(observed["ics"])
    count_by_date = dict(zip(suffix, observed["counts"], strict=True))
    maturity = []
    for session in contract.EXPECTED_MONTHLY_DATES:
        finite = count_by_date.get(session, 0)
        maturity.append(
            {
                "date": session,
                "finite_signal_count": finite,
                "eligible_signal_count": contract.SCOPE_COLUMN_COUNT,
                "scope_column_count": contract.SCOPE_COLUMN_COUNT,
                "coverage_rate": finite / contract.SCOPE_COLUMN_COUNT,
            }
        )
    evaluation = []
    maturity_by_date = {row["date"]: row for row in maturity}
    for session, common, rank_ic in zip(
        suffix,
        observed["common"],
        observed["ics"],
        strict=True,
    ):
        evaluation.append(
            {
                **copy.deepcopy(maturity_by_date[session]),
                "common_symbol_count": common,
                "rank_ic": rank_ic,
                "exclusion_reason": None,
            }
        )
    return contract.build_prior_diagnostic_attempt_v4_3(
        source_name=source_name,
        maturity_coverage_rows=maturity,
        evaluation_period_rows=evaluation,
    )


def _attempts() -> list[dict]:
    return [_attempt(str(row["source_name"])) for row in contract.ATTEMPT_SPECS]


def _distribution_inventory(distribution_name: str) -> dict:
    distribution = importlib.metadata.distribution(distribution_name)
    record_item = next(
        item
        for item in distribution.files or ()
        if str(item).endswith(".dist-info/RECORD")
    )
    record_path = Path(str(distribution.locate_file(record_item)))
    record_raw = record_path.read_bytes()
    inventory: list[dict[str, str | int]] = []
    unhashed = hash_mismatch = size_mismatch = 0
    rows = csv.reader(record_raw.decode("utf-8").splitlines())
    for relative_path, encoded_hash, recorded_size in rows:
        if not relative_path.startswith(f"{distribution_name}/"):
            continue
        if not encoded_hash.startswith("sha256="):
            unhashed += 1
            continue
        path = Path(str(distribution.locate_file(relative_path)))
        raw = path.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        encoded = encoded_hash.removeprefix("sha256=")
        padding = "=" * ((4 - len(encoded) % 4) % 4)
        record_sha = base64.urlsafe_b64decode(encoded + padding).hex()
        hash_mismatch += int(record_sha != actual_sha)
        size_mismatch += int(int(recorded_size) != len(raw))
        inventory.append(
            {"path": relative_path, "sha256": actual_sha, "size_bytes": len(raw)}
        )
    inventory.sort(key=lambda row: str(row["path"]))
    return {
        "distribution": distribution_name,
        "version": distribution.version,
        "package_prefix": f"{distribution_name}/",
        "record_path": str(record_item),
        "record_byte_sha256": hashlib.sha256(record_raw).hexdigest(),
        "record_selected_entry_count": len(inventory),
        "unhashed_selected_entry_count": unhashed,
        "hash_mismatch_count": hash_mismatch,
        "size_mismatch_count": size_mismatch,
        "file_inventory": inventory,
        "file_inventory_semantic_sha256": contract.semantic_sha256_v4_3(inventory),
    }


def _runtime_binding() -> dict:
    executable = Path(sys.executable).resolve()
    project = []
    for relative_path in contract.PROJECT_BINDING_PATHS:
        fixed = contract.FIXED_EXISTING_PROJECT_SHA256.get(relative_path)
        if fixed is None:
            raw = relative_path.encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            size = len(raw)
        else:
            raw = (PROJECT_ROOT / relative_path).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            size = len(raw)
            assert digest == fixed
        project.append(
            {"relative_path": relative_path, "byte_sha256": digest, "size_bytes": size}
        )
    return contract.build_prior_diagnostic_runtime_binding_v4_3(
        python={
            "implementation": "CPython",
            "version": "3.13.7",
            "executable": str(executable),
            "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        },
        distributions=[
            _distribution_inventory(name)
            for name, _version, _count, _sha in contract.EXPECTED_DISTRIBUTIONS
        ],
        project_bindings=project,
        source_binding=contract.SOURCE_BINDING_EXPECTED,
        matrix_bindings=contract.MATRIX_BINDINGS_EXPECTED,
    )


def _nomination(runtime_sha: str = "a" * 64) -> dict:
    return contract.build_prior_diagnostic_nomination_v4_3(
        attempts=_attempts(),
        runtime_binding_semantic_sha256=runtime_sha,
    )


def test_definition_identity_is_exact_971_byte_rebuilt_contract() -> None:
    identity = contract.build_definition_identity_payload_v4_3()
    raw = contract.canonical_json_bytes_v4_3(identity)
    assert len(raw) == 971
    assert hashlib.sha256(raw).hexdigest() == contract.DEFINITION_IDENTITY_SHA256
    assert contract.definition_identity_sha256_v4_3() == (
        "eb401bc44af71069b87eee44a3c4bb5ba73abe5337dc38a9ab1ac9e6b4bb261a"
    )
    assert contract.validate_definition_identity_payload_v4_3(identity) == identity
    assert contract.validate_definition_identity_v4_3(
        identity,
        contract.DEFINITION_IDENTITY_SHA256,
    ) == identity
    with pytest.raises(contract.FactorGovernancePriorDiagnosticNominationV4_3Error):
        contract.validate_definition_identity_v4_3(
            identity,
            "227d307ebd56ca81418e4fb8836c6aae0e41a528ff06ec2c705b5d264eab64fa",
        )

    source_drift = copy.deepcopy(identity)
    source_drift["source_binding"]["commit"] = "0" * 40
    with pytest.raises(contract.FactorGovernancePriorDiagnosticNominationV4_3Error):
        contract.validate_definition_identity_payload_v4_3(source_drift)

    payload_drift = copy.deepcopy(identity)
    payload_drift["operator_semantics"]["std_ddof"] = 0
    with pytest.raises(contract.FactorGovernancePriorDiagnosticNominationV4_3Error):
        contract.validate_definition_identity_payload_v4_3(payload_drift)


def test_attempts_recompute_exact_suffix_statistics_and_nomination_winner() -> None:
    attempts = _attempts()
    assert [len(row["maturity_coverage_rows"]) for row in attempts] == [47, 47, 47]
    assert [len(row["evaluation_period_rows"]) for row in attempts] == [38, 26, 30]
    assert [row["effective_start"] for row in attempts] == [
        "2023-04-28",
        "2024-04-30",
        "2023-12-29",
    ]
    nomination = contract.build_prior_diagnostic_nomination_v4_3(
        attempts=attempts,
        runtime_binding_semantic_sha256="b" * 64,
    )
    assert nomination["winner"]["source_name"] == "VOL_OF_VOL_20D"
    assert [row["source_name"] for row in nomination["winner"]["ranking"]] == [
        "VOL_OF_VOL_20D",
        "EXCESS_MOM_60D",
        "MOM_12M_SKIP1M",
    ]
    assert nomination["winner_candidate"]["name"] == "pv_low_vol_of_vol_20d"
    assert nomination["winner_candidate"]["initial_weight"] == 0
    assert nomination["selection_method"]["outcome_informed"] is True
    assert nomination["selection_method"]["external_label_independence"] is False
    assert all(value is False for value in nomination["authority"].values())
    assert all(value is False for value in nomination["side_effects"].values())
    assert nomination["invalid_raw_close_debug"] == contract.INVALID_RAW_CLOSE_DEBUG


def test_nomination_rejects_old_identity_row_and_row_level_tampering() -> None:
    nomination = _nomination()
    old_identity = copy.deepcopy(nomination)
    old_identity["definition_identity_sha256"] = (
        "227d307ebd56ca81418e4fb8836c6aae0e41a528ff06ec2c705b5d264eab64fa"
    )
    with pytest.raises(contract.FactorGovernancePriorDiagnosticNominationV4_3Error):
        contract.validate_prior_diagnostic_nomination_v4_3(_reseal(old_identity))

    count_tamper = copy.deepcopy(nomination)
    count_tamper["attempts"][0]["maturity_coverage_rows"][9]["finite_signal_count"] += 1
    with pytest.raises(contract.FactorGovernancePriorDiagnosticNominationV4_3Error):
        contract.validate_prior_diagnostic_nomination_v4_3(_reseal(count_tamper))

    rank_ic_tamper = copy.deepcopy(nomination)
    rank_ic_tamper["attempts"][0]["evaluation_period_rows"][0]["rank_ic"] += 0.01
    with pytest.raises(contract.FactorGovernancePriorDiagnosticNominationV4_3Error):
        contract.validate_prior_diagnostic_nomination_v4_3(_reseal(rank_ic_tamper))

    suffix_tamper = copy.deepcopy(nomination)
    suffix_tamper["attempts"][1]["evaluation_period_rows"] = suffix_tamper["attempts"][1]["evaluation_period_rows"][1:]
    with pytest.raises(contract.FactorGovernancePriorDiagnosticNominationV4_3Error):
        contract.validate_prior_diagnostic_nomination_v4_3(_reseal(suffix_tamper))


def test_runtime_record_inventory_and_crosslink_are_exact_and_tamper_evident() -> None:
    runtime = _runtime_binding()
    assert [row["record_selected_entry_count"] for row in runtime["distributions"]] == [
        889,
        1515,
        1419,
        749,
    ]
    assert [row["file_inventory_semantic_sha256"] for row in runtime["distributions"]] == [
        row[3] for row in contract.EXPECTED_DISTRIBUTIONS
    ]
    for relative_path, expected in contract.FIXED_EXISTING_PROJECT_SHA256.items():
        assert hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest() == expected

    nomination = _nomination(runtime["artifact_semantic_sha256"])
    combined = contract.validate_prior_diagnostic_nomination_against_runtime_v4_3(
        nomination,
        runtime,
    )
    assert combined["nomination"]["run_id"] == combined["runtime_binding"]["run_id"]

    inventory_tamper = copy.deepcopy(runtime)
    inventory_tamper["distributions"][0]["file_inventory"][0]["size_bytes"] += 1
    with pytest.raises(contract.FactorGovernancePriorDiagnosticNominationV4_3Error):
        contract.validate_prior_diagnostic_runtime_binding_v4_3(_reseal(inventory_tamper))

    source_tamper = copy.deepcopy(runtime)
    source_tamper["source_binding"]["cutoff_date"] = "2026-07-16"
    with pytest.raises(contract.FactorGovernancePriorDiagnosticNominationV4_3Error):
        contract.validate_prior_diagnostic_runtime_binding_v4_3(_reseal(source_tamper))

    crosslink_tamper = copy.deepcopy(nomination)
    crosslink_tamper["runtime_binding_semantic_sha256"] = "f" * 64
    with pytest.raises(contract.FactorGovernancePriorDiagnosticNominationV4_3Error):
        contract.validate_prior_diagnostic_nomination_against_runtime_v4_3(
            _reseal(crosslink_tamper),
            runtime,
        )


def test_canonical_helpers_reject_nonfinite_values() -> None:
    with pytest.raises(contract.FactorGovernancePriorDiagnosticNominationV4_3Error):
        contract.canonical_json_bytes_v4_3({"value": math.nan})
    assert contract.canonical_file_bytes_v4_3({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'
