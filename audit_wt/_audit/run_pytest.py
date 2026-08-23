"""Run the repo's pytest suite from the correct working dir."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import pytest

sys.exit(pytest.main(sys.argv[1:] or ["tests", "-q", "--no-header"]))