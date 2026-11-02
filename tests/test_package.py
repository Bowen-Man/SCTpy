from sctransform_native import __version__


def test_version_is_alpha_release() -> None:
    assert __version__ == "0.1.0a1"
