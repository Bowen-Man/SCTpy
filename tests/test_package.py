import subprocess
import sys

from sctpy import __version__


def test_version_is_alpha_release() -> None:
    assert __version__ == "0.1.0a1"


def test_cpu_package_import_does_not_import_cupy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, sctpy; assert 'cupy' not in sys.modules",
        ],
        check=False,
    )
    assert completed.returncode == 0
