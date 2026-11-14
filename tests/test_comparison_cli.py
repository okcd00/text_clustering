import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_setup_errors_are_actionable():
 r=subprocess.run(['bash',str(ROOT/'scripts/evaluate_static.sh')],env={'PATH':os.environ['PATH']},capture_output=True,text=True)
 assert r.returncode and 'SUITE_PYTHON' in r.stderr

def test_missing_required_paths():
 r=subprocess.run([sys.executable,str(ROOT/'scripts/run_comparison_suite.py')],capture_output=True,text=True)
 assert r.returncode and '--cache, --output and --work required' in r.stderr
