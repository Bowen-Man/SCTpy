import os
import sys

import pytest

pytest.importorskip("psutil")
from benchmarks.run_baseline import _run_budgeted


@pytest.mark.parametrize(
    "seconds,memory,reason", [(0.05, 1024**3, "time"), (5, 1, "memory")]
)
def test_budget_terminates_worker(seconds, memory, reason):
    with pytest.raises(RuntimeError, match=f"{reason} budget exceeded"):
        _run_budgeted(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            os.environ.copy(),
            seconds,
            memory,
        )
