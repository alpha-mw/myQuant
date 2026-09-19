"""One isolated freeze/install after the reviewed canonical scan repair."""
from datetime import datetime, timezone
import json,os,sys
from pathlib import Path
source=Path('/Users/maxwell/mySpace/myQuant')
location=source/'.agent/acceptance/phase15-canonical-scan-location.json'
if location.exists():raise SystemExit('Candidate already recorded; inspect, do not duplicate')
root=Path('/private/tmp/myquant-canonical-scan-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
state={'root':str(root),'state':'BUILDING','pid':os.getpid(),'production_deployed':False}
location.write_text(json.dumps(state,indent=2)+'\n')
os.environ['UV_OFFLINE']='1';os.environ['UV_CACHE_DIR']='/Users/maxwell/.cache/uv'
sys.path.insert(0,str(source/'tests/unit'))
from _native_daily_release_fixture import prepare_synthetic_release
try:
 r=prepare_synthetic_release(source,root)
 state.update(state='INSTALLED_VERIFIED',commit=r['commit'],import_origin=r['runtime_verification']['import_origin'])
except BaseException as exc:
 state.update(state='FAIL',error_type=type(exc).__name__,error=str(exc));location.write_text(json.dumps(state,indent=2)+'\n');raise
state['finished_at']=datetime.now(timezone.utc).isoformat();location.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
