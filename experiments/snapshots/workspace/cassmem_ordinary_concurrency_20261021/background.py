"""Owned hidden background controller; UI/chat lifetime is independent."""
import sys
from pathlib import Path
out=Path(__file__).resolve().parent/'run_v1'
log=(out/'runner.log').open('a',encoding='utf-8',buffering=1)
sys.stdout=log;sys.stderr=log
import runner
runner.campaign()
