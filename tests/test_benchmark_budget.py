import json
import os
import subprocess
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


def test_profiler_preserves_cumulative_call_time(tmp_path):
    fixture = tmp_path / "fixture.npz"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.generate_fixture",
            "--output",
            str(fixture),
            "--genes",
            "100",
            "--cells",
            "60",
        ],
        check=True,
        capture_output=True,
    )
    report = tmp_path / "profile.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.profile_stages",
            "--fixture",
            str(fixture),
            "--manifest",
            str(fixture.with_suffix(".selection.json")),
            "--output",
            str(report),
            "--profile",
            str(tmp_path / "profile.prof"),
        ],
        check=True,
        capture_output=True,
    )
    data = json.loads(report.read_text())
    fit = next(
        row
        for row in data["top_cprofile_cumulative"]
        if row["function"] == "fit_transform"
    )
    assert fit["cumulative_seconds"] >= 0.8 * data["wall_seconds"]
    assert data["peak_rss_bytes"] >= data["baseline_rss_bytes"]
