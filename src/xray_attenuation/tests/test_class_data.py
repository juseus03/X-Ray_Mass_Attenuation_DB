from data import Data


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
