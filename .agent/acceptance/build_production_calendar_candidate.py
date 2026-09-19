"""Freeze current authorized source and verify one isolated offline installation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

source=Path('/Users/maxwell/mySpace/myQuant')
root=Path('/private/tmp/myquant-production-calendar-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
location=source/'.agent/acceptance/phase12-production-calendar-location.json'
if location.exists():
 raise SystemExit('Candidate location already exists; inspect it instead of duplicating build')
state={'root':str(root),'state':'BUILDING','pid':os.getpid(),'production_deployed':False,'started_at':datetime.now(timezone.utc).isoformat()}
location.write_text(json.dumps(state,indent=2)+'\n')
cache=Path('/Users/maxwell/.cache/uv')
assert cache.is_dir()
os.environ['UV_CACHE_DIR']=str(cache)
os.environ['UV_OFFLINE']='1'
sys.path.insert(0,str(source/'tests/unit'))
from _native_daily_release_fixture import prepare_synthetic_release
try:
 receipt=prepare_synthetic_release(source,root)
 state.update(state='INSTALLED_VERIFIED',commit=receipt['commit'],import_origin=receipt['runtime_verification']['import_origin'],finished_at=datetime.now(timezone.utc).isoformat())
except BaseException as exc:
 state.update(state='FAIL',error_type=type(exc).__name__,error=str(exc),finished_at=datetime.now(timezone.utc).isoformat())
 location.write_text(json.dumps(state,indent=2)+'\n')
 raise
location.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state),flush=True)
