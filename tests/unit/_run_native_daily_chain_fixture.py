"""Bounded fresh native integration case; stops on first unsuccessful process.

Run once, then follow its actual process handle. Never rerun to recover a timed-out
observation; individual native receipt/recovery boundaries must be inspected first.
"""

import json
from pathlib import Path
import subprocess
import sys


def run(source: Path, root: Path) -> dict:
    # Script invocation otherwise follows the editable main-workspace install.
    sys.path.insert(0, str(source))
    from _native_daily_release_fixture import prepare_synthetic_release

    release = prepare_synthetic_release(source, root)
    python = release["python"]
    helper = source / "tests/unit"
    steps = [
        (
            "calendar-baseline",
            "_native_daily_calendar_fixture.py",
            [str(root), str(helper / "test_tushare_calendar_authority.py"), "2026-08-24"],
        ),
        (
            "factor-prepare",
            "_native_daily_factor_prepare_fixture.py",
            [str(root), "--create-market"],
        ),
        ("factor-activate", "_native_daily_factor_activation_fixture.py", [str(root), "activate"]),
        ("successor-market", "_native_daily_rollover_fixture.py", [str(root)]),
        (
            "calendar-successor",
            "_native_daily_calendar_fixture.py",
            [str(root), str(helper / "test_tushare_calendar_authority.py"), "2026-08-25"],
        ),
        ("maintenance", "_native_daily_maintenance_fixture.py", [str(root)]),
        ("factor-rollover", "_native_daily_factor_activation_fixture.py", [str(root), "rollover"]),
        ("factor-observe", "_native_daily_factor_activation_fixture.py", [str(root), "observe"]),
        ("factor-core", "_native_daily_factor_activation_fixture.py", [str(root), "core"]),
    ]
    completed = []
    for name, filename, arguments in steps:
        print("START", name, flush=True)
        result = subprocess.run(
            [python, "-I", str(helper / filename), *arguments],
            cwd=root,
            text=True,
            capture_output=True,
        )
        (root / (name + ".stdout.log")).write_text(result.stdout)
        (root / (name + ".stderr.log")).write_text(result.stderr)
        if result.returncode:
            state = {
                "state": "FAILED",
                "failed_step": name,
                "completed": completed,
                "exit_code": result.returncode,
                "full_dag_proof": False,
            }
            (root / "chain-state.json").write_text(json.dumps(state, indent=2) + "\n")
            print(result.stderr[-2500:], flush=True)
            raise RuntimeError("native fixture stopped at " + name)
        value = json.loads(result.stdout)
        (root / (name + "-result.json")).write_text(json.dumps(value, indent=2) + "\n")
        completed.append(name)
        (root / "chain-state.json").write_text(
            json.dumps(
                {"state": "RUNNING", "completed": completed, "full_dag_proof": False}, indent=2
            )
            + "\n"
        )
        print("PASS", name, flush=True)
    final = {
        "state": "COMPLETED_NATIVE_CORE_CASE",
        "completed": completed,
        "synthetic": True,
        "full_dag_proof": False,
    }
    (root / "chain-state.json").write_text(json.dumps(final, indent=2) + "\n")
    return final


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
