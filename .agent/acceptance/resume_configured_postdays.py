"""Resume only aggregate verification after the retained lock-guard failure."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path("/Users/maxwell/mySpace/myQuant")
ROOT = Path(sys.argv[1]).resolve(strict=True)
status_path = ROOT / "configured-postdays-resume-status.json"
if status_path.exists():
    raise ValueError("postdays continuation already exists")
old_path = ROOT / "configured-successors-status.json"
old_raw = old_path.read_bytes()
old = json.loads(old_raw)
expected = ["repeat", "20260828", "20260831", "20260901", "20260902", "five-day-replay"]
assert old["state"] == "FAIL" and not old["helper_drift"]
assert [x["stage"] for x in old["steps"]] == expected
assert [x["exit_code"] for x in old["steps"]] == [0, 0, 0, 0, 0, 1]
assert (
    "historical replay attempted writer lock"
    in (ROOT / "configured-successor-five-day-replay.log").read_text()
)
receipt = json.loads((ROOT / "fixture-receipt.json").read_bytes())
old_manifest = json.loads((ROOT / "successor-validation-helper-manifest.json").read_bytes())
helpers = ROOT / "postdays-validation-helpers"
helpers.mkdir(mode=0o700)
pins = {}
for name, digest in old_manifest["helper_sha256"].items():
    raw = (ROOT / "successor-validation-helpers" / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    if name == "_verify_five_native_days.py":
        raw = (SOURCE / "tests/unit" / name).read_bytes()
    (helpers / name).write_bytes(raw)
    (helpers / name).chmod(0o600)
    pins[name] = hashlib.sha256(raw).hexdigest()
# Extract the original stage body, without executing its launcher.
tree = ast.parse((ROOT / "acceptance-drivers/run_configured_successors.py").read_text())
program = next(
    ast.literal_eval(n.value)
    for n in tree.body
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "program" for t in n.targets)
)
program = program.replace(
    "acceptance_harness_interruption_disclosed=resume,continuous_uninterrupted_harness_claimed=not resume",
    "acceptance_harness_interruption_disclosed=True,continuous_uninterrupted_harness_claimed=False",
)
# Preserve the exact failure and all completed daily proof bytes.
proof_files = [ROOT / "configured-native-proof.json"] + [
    ROOT / ("configured-native-proof-" + d + ".json") for d in expected[1:5]
]
proof_pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in proof_files}
manifest = {
    "helper_sha256": pins,
    "original_helper_manifest": old_manifest,
    "previous_failure_ref": {"path": str(old_path), "sha256": hashlib.sha256(old_raw).hexdigest()},
    "daily_proof_sha256": proof_pins,
    "runtime_commit": receipt["commit"],
    "only_changed_helper": "_verify_five_native_days.py",
    "production_deployed": False,
}
(ROOT / "postdays-validation-helper-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
state = {
    "state": "RUNNING",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "steps": [],
    "previous_failure_ref": manifest["previous_failure_ref"],
    "acceptance_harness_interruption_disclosed": True,
    "production_deployed": False,
}
environment = dict(os.environ)
environment.pop("PYTHONPATH", None)
for stage in ("five-day-replay", "morning", "top100-recovery", "closed"):
    assert all(hashlib.sha256((helpers / n).read_bytes()).hexdigest() == h for n, h in pins.items())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in proof_pins.items())
    log = ROOT / ("configured-successor-resumed-" + stage + ".log")
    assert not log.exists()
    started = time.monotonic()
    with log.open("x") as stream:
        child = subprocess.Popen(
            [
                receipt["python"],
                "-I",
                "-c",
                program,
                str(helpers),
                str(ROOT),
                str(SOURCE / ".venv/lib/python3.13/site-packages"),
                stage,
                "resume-prepared",
            ],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        state.update(active_step=stage, child_pid=child.pid)
        status_path.write_text(json.dumps(state, indent=2) + "\n")
        code = child.wait()
    state["steps"].append(
        {
            "stage": stage,
            "exit_code": code,
            "seconds": round(time.monotonic() - started, 2),
            "log": str(log),
        }
    )
    if code:
        state["state"] = "FAIL"
        break
else:
    state["state"] = "PASS"
state.pop("child_pid", None)
state.pop("active_step", None)
state["finished_at"] = datetime.now(timezone.utc).isoformat()
state["helper_drift"] = [
    n for n, h in pins.items() if hashlib.sha256((helpers / n).read_bytes()).hexdigest() != h
]
status_path.write_text(json.dumps(state, indent=2) + "\n")
