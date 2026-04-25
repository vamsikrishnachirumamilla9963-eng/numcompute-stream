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
        