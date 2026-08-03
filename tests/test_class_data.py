import pytest

from xray_attenuation.data import Data


class TestDataClass:

    data = Data()

    def test_elements_df(self):
        assert self.data.df_elements.collect().shape == (1971, 93)

    def test_elements_name_df(self):
        assert self.data.df_elements_names.shape == (92, 6)

    def test_compounds_df(self):
        assert self.data.df_compounds.collect().shape == (1971, 18)

    def test_compounds_name_df(self):
        assert self.data.df_compounds_names.shape == (17, 4)

    def test_spectra_df(self):
        assert self.data.df_spectra.shape == (1000, 93)

    def test_resolve_element_name(self):
        assert self.data.resolve_material_name("Aluminum") == ("Aluminum", False)

    def test_resolve_element_name_any_case(self):
        assert self.data.resolve_material_name("aluminum") == ("Aluminum", False)

    def test_resolve_element_symbol(self):
        assert self.data.resolve_material_name("Al") == ("Aluminum", False)

    def test_resolve_element_symbol_any_case(self):
        assert self.data.resolve_material_name("al") == ("Aluminum", False)

    def test_resolve_compound_name(self):
        assert self.data.resolve_material_name("Cadmium Telluride") == (
            "Cadmium Telluride",
            True,
        )

    def test_resolve_compound_name_any_case(self):
        assert self.data.resolve_material_name("cadmium telluride") == (
            "Cadmium Telluride",
            True,
        )

    def test_resolve_strips_whitespace(self):
        assert self.data.resolve_material_name("  Aluminum  ") == ("Aluminum", False)

    def test_resolve_unknown_material(self):
        assert self.data.resolve_material_name("Kryptonite") is None

    def test_resolve_empty_name(self):
        assert self.data.resolve_material_name("") is None

    def test_get_linear_attenuation(self):
        mu = self.data.get_linear_attenuation("Aluminum", 10.0)
        assert mu == pytest.approx(71.61916827)

    def test_get_linear_attenuation_energy_not_found(self):
        mu = self.data.get_linear_attenuation("Aluminum", 2.0)
        assert mu is None

    def test_get_linear_attenuation_element_not_found(self):
        mu = self.data.get_linear_attenuation("Kryptonite", 100.0)
        assert mu is None

    def test_get_linear_attenuation_compound(self):
        mu = self.data.get_linear_attenuation(
            "Cadmium Telluride", 20.0, is_compound=True
        )
        assert mu == pytest.approx(138.9626446)

    def test_get_linear_attenuation_not_compound(self):
        mu = self.data.get_linear_attenuation(
            "Cadmium Telluride", 20.0, is_compound=False
        )
        assert mu is None

    def test_get_linear_attenuation_compound_not_found(self):
        mu = self.data.get_linear_attenuation("Kryptonite", 20.0, is_compound=True)
        assert mu is None
