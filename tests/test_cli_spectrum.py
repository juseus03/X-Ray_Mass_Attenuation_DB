import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from xray_attenuation.cli import CLI
from xray_attenuation.data import Data

# Energy to probe the spectra at. Well inside the 3-200 keV attenuation grid and far
# enough above the 1e-35 sentinel that no clamping interferes with the comparisons.
PROBE_KEV = 40.0
TUBE_KV = 60


@pytest.fixture(scope="module")
def shared_data():
    """One Data instance for the module: loading it is the expensive part and every
    lookup below is read-only"""
    return Data()


@pytest.fixture
def cli(shared_data, monkeypatch):
    """A fresh CLI per test, but reusing the module-scoped database"""
    monkeypatch.setattr("xray_attenuation.cli.Data", lambda: shared_data)
    return CLI(is_full_spectrum=True)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def value_at(cli, column, energy=PROBE_KEV):
    """Reads one spectrum column at one energy"""
    return cli.spectrum_df.filter(pl.col("Energy[keV]") == energy)[column][0]


class TestAddBaseSpectrum:
    def test_selects_the_requested_tube_voltage(self, cli):
        cli.add_base_spectrum(TUBE_KV)

        assert cli.max_kv == "60"
        assert cli.spectrum_df.columns == ["Energy[keV]", "60"]
        assert cli.spectrum_df.height == 1000

    def test_integer_voltage_is_not_reported_as_snapped(self, cli, capsys):
        cli.add_base_spectrum(TUBE_KV)

        assert "NOTICE" not in capsys.readouterr().out

    def test_non_integer_voltage_snaps_and_says_so(self, cli, capsys):
        cli.add_base_spectrum(60.4)

        assert cli.max_kv == "60"
        out = capsys.readouterr().out
        assert "NOTICE" in out
        assert "60.4 kV" in out

    def test_snapping_is_half_to_even(self, cli):
        # numpy rounds halfway cases to even, so these two land on the same side
        cli.add_base_spectrum(60.5)
        assert cli.max_kv == "60"

        cli.add_base_spectrum(61.5)
        assert cli.max_kv == "62"


class TestAddFilter:
    def test_single_filter_follows_beer_lambert(self, cli, shared_data):
        cli.add_base_spectrum(TUBE_KV)
        base = value_at(cli, "60")

        cli.add_filter("Aluminum", TUBE_KV, 0.1, False)

        mu_al = shared_data.get_linear_attenuation("Aluminum", PROBE_KEV)
        assert value_at(cli, "1_Aluminum_0.1cm") == pytest.approx(
            base * np.exp(-mu_al * 0.1)
        )

    def test_filters_stack_cumulatively(self, cli, shared_data):
        cli.add_base_spectrum(TUBE_KV)
        base = value_at(cli, "60")

        cli.add_filter("Aluminum", TUBE_KV, 0.1, False)
        cli.add_filter("Copper", TUBE_KV, 0.1, False)

        mu_al = shared_data.get_linear_attenuation("Aluminum", PROBE_KEV)
        mu_cu = shared_data.get_linear_attenuation("Copper", PROBE_KEV)

        stacked = base * np.exp(-mu_al * 0.1) * np.exp(-mu_cu * 0.1)
        assert value_at(cli, "2_Copper_0.1cm") == pytest.approx(stacked)

    def test_second_filter_is_not_applied_to_the_bare_spectrum(self, cli, shared_data):
        """A filter's column holds the whole stack, not that filter alone"""
        cli.add_base_spectrum(TUBE_KV)
        base = value_at(cli, "60")

        cli.add_filter("Aluminum", TUBE_KV, 0.1, False)
        cli.add_filter("Copper", TUBE_KV, 0.1, False)

        mu_cu = shared_data.get_linear_attenuation("Copper", PROBE_KEV)
        copper_alone = base * np.exp(-mu_cu * 0.1)

        assert value_at(cli, "2_Copper_0.1cm") < copper_alone

    def test_repeated_material_stacks_rather_than_replacing(self, cli, shared_data):
        cli.add_base_spectrum(TUBE_KV)
        base = value_at(cli, "60")

        cli.add_filter("Aluminum", TUBE_KV, 0.1, False)
        cli.add_filter("Aluminum", TUBE_KV, 0.2, False)

        mu_al = shared_data.get_linear_attenuation("Aluminum", PROBE_KEV)
        # 0.1 cm then 0.2 cm of the same material is 0.3 cm of it
        assert value_at(cli, "2_Aluminum_0.2cm") == pytest.approx(
            base * np.exp(-mu_al * 0.3)
        )

    def test_compound_filter(self, cli, shared_data):
        cli.add_base_spectrum(TUBE_KV)
        base = value_at(cli, "60")

        cli.add_filter("Cadmium Telluride", TUBE_KV, 0.01, True)

        mu = shared_data.get_linear_attenuation(
            "Cadmium Telluride", PROBE_KEV, is_compound=True
        )
        assert value_at(cli, "1_Cadmium Telluride_0.01cm") == pytest.approx(
            base * np.exp(-mu * 0.01)
        )

    @pytest.mark.parametrize("thickness", [0.0, -0.5])
    def test_non_positive_thickness_is_rejected(self, cli, thickness, capsys):
        cli.add_base_spectrum(TUBE_KV)

        cli.add_filter("Aluminum", TUBE_KV, thickness, False)

        assert cli.filters == []
        assert cli.spectrum_df.columns == ["Energy[keV]", "60"]
        assert "WARNING" in capsys.readouterr().out

    def test_join_trims_energies_below_the_attenuation_grid(self, cli):
        cli.add_base_spectrum(TUBE_KV)
        assert cli.spectrum_df.height == 1000

        cli.add_filter("Aluminum", TUBE_KV, 0.1, False)

        # The NIST tables start at 3 keV, so 0.1-2.9 keV drops out. Intentional
        assert cli.spectrum_df.height == 971
        assert cli.spectrum_df["Energy[keV]"].min() == pytest.approx(3.0)


def build_stack(cli, materials):
    cli.add_base_spectrum(TUBE_KV)
    for name, thickness in materials:
        cli.add_filter(name, TUBE_KV, thickness, False)
    return cli


class TestRemoveFilter:
    STACK = [("Aluminum", 0.1), ("Copper", 0.1), ("Iron", 0.1)]

    def test_removing_the_last_filter(self, cli):
        build_stack(cli, self.STACK)

        cli.remove_filter(2)

        assert [f.name for f in cli.filters] == ["Aluminum", "Copper"]
        assert cli.spectrum_df.columns == [
            "Energy[keV]",
            "60",
            "1_Aluminum_0.1cm",
            "2_Copper_0.1cm",
        ]

    def test_removing_a_middle_filter_recomputes_and_renumbers(self, cli):
        build_stack(cli, self.STACK)

        cli.remove_filter(0)

        assert [f.name for f in cli.filters] == ["Copper", "Iron"]
        assert cli.spectrum_df.columns == [
            "Energy[keV]",
            "60",
            "1_Copper_0.1cm",
            "2_Iron_0.1cm",
        ]

    def test_recomputed_stack_matches_building_it_directly(self, cli, shared_data):
        """Removing Aluminum from Al->Cu->Fe must equal a fresh Cu->Fe stack"""
        build_stack(cli, self.STACK)
        cli.remove_filter(0)
        recomputed = value_at(cli, "2_Iron_0.1cm")

        direct = CLI(is_full_spectrum=True)
        direct.data = shared_data
        build_stack(direct, [("Copper", 0.1), ("Iron", 0.1)])

        assert recomputed == pytest.approx(value_at(direct, "2_Iron_0.1cm"))

    @pytest.mark.parametrize("index", [-1, 3, 99])
    def test_out_of_range_index_is_a_no_op(self, cli, index, capsys):
        build_stack(cli, self.STACK)
        before = cli.spectrum_df.columns.copy()

        cli.remove_filter(index)

        assert len(cli.filters) == 3
        assert cli.spectrum_df.columns == before
        assert "ERROR" in capsys.readouterr().out


class TestPlotSpectra:
    def test_plots_one_line_per_spectrum_with_cumulative_labels(self, cli):
        build_stack(cli, [("Aluminum", 0.1), ("Copper", 0.2)])

        cli.plot_spectra()

        ax = plt.gcf().axes[0]
        assert [line.get_label() for line in ax.get_lines()] == [
            "60 kV",
            "+ Aluminum 0.1 cm",
            "+ Copper 0.2 cm",
        ]

    def test_plots_the_bare_spectrum_when_there_are_no_filters(self, cli):
        cli.add_base_spectrum(TUBE_KV)

        cli.plot_spectra()

        ax = plt.gcf().axes[0]
        assert [line.get_label() for line in ax.get_lines()] == ["60 kV"]

    def test_axes_are_labelled_and_log_scaled(self, cli):
        build_stack(cli, [("Aluminum", 0.1)])

        cli.plot_spectra()

        ax = plt.gcf().axes[0]
        assert ax.get_xlabel() == "Energy [keV]"
        assert ax.get_ylabel() == "Intensity [a.u.]"
        assert ax.get_yscale() == "log"

    def test_style_does_not_leak_into_global_rcparams(self, cli):
        import matplotlib as mpl

        before = mpl.rcParams["font.size"]
        build_stack(cli, [("Aluminum", 0.1)])

        cli.plot_spectra()

        assert mpl.rcParams["font.size"] == before

    def test_without_a_base_spectrum_it_reports_instead_of_crashing(self, cli, capsys):
        cli.plot_spectra()

        assert "ERROR" in capsys.readouterr().out

    def test_save_plot_writes_a_png(self, cli, tmp_path, monkeypatch):
        monkeypatch.setattr("xray_attenuation.cli.TMP_PATH", tmp_path)
        cli.save_plot = True
        build_stack(cli, [("Aluminum", 0.1)])

        cli.plot_spectra()

        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert written[0].stat().st_size > 0

    def test_save_spectrum(self, cli):

        import tempfile
        from pathlib import Path

        base_path = Path(tempfile.gettempdir())

        assert cli.save_spectrum(base_path) is False

        build_stack(cli, [("Aluminum", 0.1), ("Lead", 0.2)])

        assert cli.save_spectrum(base_path / "test.txt") is False
        assert cli.save_spectrum(base_path / "test.csv") is True

        data = pl.read_csv(base_path / "test.csv")

        assert data.shape == (971, 4)
        assert data.columns == ["Energy[keV]", "60", "1_Aluminum_0.1cm", "2_Lead_0.2cm"]


class TestPhysicsMetrics:
    def test_total_filtered_franction(self, cli):

        assert cli.get_total_filtered_fraction() is None

        cli.add_base_spectrum(TUBE_KV)

        cli.spectrum_df = cli.spectrum_df.with_columns(
            (pl.col(str(TUBE_KV)) * 0.5).alias("frac")
        )
        assert cli.get_total_filtered_fraction() == pytest.approx(0.5, abs=1e-5)

        cli.spectrum_df = cli.spectrum_df.with_columns(
            (pl.col(str(TUBE_KV)) * 0.1).alias("frac")
        )
        assert cli.get_total_filtered_fraction() == pytest.approx(0.1, abs=1e-5)

        cli.spectrum_df = cli.spectrum_df.with_columns(
            (pl.col(str(TUBE_KV)) * 0.8).alias("frac")
        )
        assert cli.get_total_filtered_fraction() == pytest.approx(0.8, abs=1e-5)

        cli.spectrum_df = cli.spectrum_df.with_columns(
            (pl.col(str(TUBE_KV)) * 0.8).alias("frac")
        )
        assert cli.get_total_filtered_fraction() == pytest.approx(0.8, abs=1e-5)

        with pytest.raises(ValueError):
            cli.spectrum_df = cli.spectrum_df.with_columns(
                (pl.col(str(TUBE_KV)) * 1.8).alias("frac")
            )
            cli.get_total_filtered_fraction()

    def test_mean_energy_spectrum(self, cli):

        assert cli.get_mean_energy_spectrum() is None

        voltages = [50, 60, 70, 80, 90, 100]
        answ = []

        for v in voltages:
            cli.add_base_spectrum(v)
            answ.append(cli.get_mean_energy_spectrum())

        assert all(f > 0 for f in answ)
        assert all(v > f for f, v in zip(answ, voltages, strict=True))
        assert answ == sorted(answ)

    def test_get_hvl(self, cli):
        assert cli.get_hvl() is None

        cli.add_base_spectrum(TUBE_KV)
        hvl = cli.get_hvl()

        cli.add_filter("Aluminum", TUBE_KV, hvl * 1e-1, False)
        frac = cli.get_total_filtered_fraction()

        assert frac == pytest.approx(0.5, abs=1e-4)

    def test_get_effective_energy(self, cli):

        assert cli.get_effective_energy() is None

        cli.add_base_spectrum(TUBE_KV)
        answ = []

        answ.append(cli.get_effective_energy())

        cli.add_filter("Aluminum", TUBE_KV, 0.1, False)
        answ.append(cli.get_effective_energy())

        cli.add_filter("Aluminum", TUBE_KV, 0.1, False)
        answ.append(cli.get_effective_energy())

        assert answ == sorted(answ)
