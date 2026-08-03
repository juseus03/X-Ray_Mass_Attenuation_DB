import tempfile
from pathlib import Path

from xray_attenuation.cli import CLI, Filter, _sanitize_for_filename


def test_sanitize_for_filename_leaves_simple_names_alone():
    assert _sanitize_for_filename("Al") == "Al"
    assert _sanitize_for_filename("Aluminum") == "Aluminum"


def test_sanitize_for_filename_cleans_compound_names():
    assert _sanitize_for_filename("Bone, Cortical (ICRU-44)") == "Bone_Cortical_ICRU-44"
    assert (
        _sanitize_for_filename("Brain, Grey/White Matter (ICRU-44)")
        == "Brain_Grey_White_Matter_ICRU-44"
    )
    assert _sanitize_for_filename("Water, Liquid") == "Water_Liquid"


class _StubCLI:
    """Minimal stand-in exposing only what build_plot_path reads, so the test does
    not have to load the whole database"""

    def __init__(self, max_kv, filters):
        self.max_kv = max_kv
        self.filters = filters


def test_build_plot_path_names_the_kv_and_the_filters():
    stub = _StubCLI(
        "100",
        [
            Filter("Aluminum", 0.1, False),
            Filter("Bone, Cortical (ICRU-44)", 0.05, True),
        ],
    )

    path = CLI.build_plot_path(stub)

    assert path.parent == Path(tempfile.gettempdir())
    assert path.suffix == ".png"
    assert path.name.startswith(
        "xray_spectrum_100kV_Aluminum-0.1cm_Bone_Cortical_ICRU-44-0.05cm_"
    )


def test_build_plot_path_stays_within_the_filename_limit():
    mylar = Filter("Polyethylene Terephthalate, (Mylar)", 0.1, True)
    stub = _StubCLI("100", [mylar] * 20)

    path = CLI.build_plot_path(stub)

    assert len(path.name) < 255
    assert path.name.startswith("xray_spectrum_100kV_20filters_")
