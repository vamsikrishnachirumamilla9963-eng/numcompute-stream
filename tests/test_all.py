import sys, os, tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute import io as nc_io

class TestLoadCSV:
    def test_basic_load(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n"); fname = f.name
        data = nc_io.load_csv(fname, skip_header=True)
        os.unlink(fname)
        assert data.shape == (2, 3)
        assert np.allclose(data[0], [1, 2, 3])

    def test_missing_values_become_nan(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("a,b\n1,\n3,4\n"); fname = f.name
        data = nc_io.load_csv(fname, skip_header=True)
        os.unlink(fname)
        assert np.isnan(data[0, 1])

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("a,b\n"); fname = f.name
        with pytest.raises(ValueError):
            nc_io.load_csv(fname, skip_header=True)
        os.unlink(fname)

    def test_save_and_reload(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            fname = f.name
        nc_io.save_csv(X, fname, header='a,b')
        loaded = nc_io.load_csv(fname, skip_header=True)
        os.unlink(fname)
        assert np.allclose(loaded, X)

    def test_save_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            nc_io.save_csv(np.ones((2, 2, 2)), "/tmp/bad.csv")
    
class TestLoadCSVChunks:
    def test_chunks_cover_all_rows(self):
        rows = '\n'.join(f'{i},{i*2}' for i in range(25))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('a,b\n' + rows + '\n'); fname = f.name
        chunks = list(nc_io.load_csv_chunks(fname, chunk_size=10))
        os.unlink(fname)
        assert sum(len(c) for c in chunks) == 25
        assert len(chunks) == 3

    def test_tab_delimiter(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write('1\t2\t3\n4\t5\t6\n'); fname = f.name
        data = nc_io.load_csv(fname, delimiter='\t', skip_header=False)
        os.unlink(fname)
        assert data.shape == (2, 3)

from numcompute.preprocessing import StandardScaler

class TestStandardScaler:
    def test_zero_mean_unit_std(self):
        X = np.array([[1., 2.], [3., 4.], [5., 6.]])
        Xt = StandardScaler().fit_transform(X)
        assert np.allclose(Xt.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(Xt.std(axis=0),  1, atol=1e-10)

    def test_zero_variance_feature(self):
        X = np.array([[1., 5.], [1., 6.], [1., 7.]])
        Xt = StandardScaler().fit_transform(X)
        assert np.allclose(Xt[:, 0], 0)

    def test_inverse_transform(self):
        X = np.array([[1., 2.], [3., 4.]])
        sc = StandardScaler().fit(X)
        assert np.allclose(sc.inverse_transform(sc.transform(X)), X, atol=1e-10)

    def test_nan_preserved(self):
        X = np.array([[1., np.nan], [3., 4.], [5., 6.]])
        Xt = StandardScaler().fit_transform(X)
        assert np.isnan(Xt[0, 1])

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            StandardScaler().transform(np.array([[1., 2.]]))

from numcompute.preprocessing import MinMaxScaler

class TestMinMaxScaler:
    def test_range_01(self):
        X = np.array([[0.], [5.], [10.]])
        Xt = MinMaxScaler().fit_transform(X)
        assert np.isclose(Xt.min(), 0.) and np.isclose(Xt.max(), 1.)

    def test_custom_range(self):
        X = np.array([[0.], [10.]])
        Xt = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
        assert np.isclose(Xt[0, 0], -1.) and np.isclose(Xt[1, 0], 1.)

    def test_constant_feature(self):
        X = np.array([[3., 1.], [3., 2.]])
        Xt = MinMaxScaler().fit_transform(X)
        assert np.allclose(Xt[:, 0], 0.)

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            MinMaxScaler(feature_range=(5, 1))

from numcompute.preprocessing import SimpleImputer

class TestSimpleImputer:
    def test_mean_imputation(self):
        X = np.array([[1., np.nan], [3., 4.]])
        Xt = SimpleImputer(strategy="mean").fit_transform(X)
        assert np.isclose(Xt[0, 1], 4.0)

    def test_constant_imputation(self):
        X = np.array([[np.nan, 2.], [3., np.nan]])
        Xt = SimpleImputer(strategy="constant", fill_value=-999).fit_transform(X)
        assert np.isclose(Xt[0, 0], -999.) and np.isclose(Xt[1, 1], -999.)

    def test_median_imputation(self):
        X = np.array([[1.], [3.], [np.nan], [5.]])
        Xt = SimpleImputer(strategy="median").fit_transform(X)
        assert np.isclose(Xt[2, 0], np.median([1, 3, 5]))

    def test_all_nan_column_stays_nan(self):
        X = np.array([[np.nan], [np.nan]])
        Xt = SimpleImputer(strategy="mean").fit_transform(X)
        assert np.all(np.isnan(Xt))

from numcompute.preprocessing import OneHotEncoder

class TestOneHotEncoder:
    def test_basic_encoding(self):
        X = np.array([[0.], [1.], [2.], [0.]])
        Xt = OneHotEncoder().fit_transform(X)
        assert Xt.shape == (4, 3)
        assert np.allclose(Xt.sum(axis=1), 1)

    def test_multi_column(self):
        X = np.array([[0., 1.], [1., 0.]])
        Xt = OneHotEncoder().fit_transform(X)
        assert Xt.shape == (2, 4)

    def test_unknown_category_all_zeros(self):
        ohe = OneHotEncoder().fit(np.array([[0.], [1.], [2.]]))
        Xt = ohe.transform(np.array([[99.]]))
        assert np.all(Xt == 0)

