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
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            fname = f.name
        nc_io.save_csv(x, fname, header='a,b')
        loaded = nc_io.load_csv(fname, skip_header=True)
        os.unlink(fname)
        assert np.allclose(loaded, x)

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
        x = np.array([[1., 2.], [3., 4.], [5., 6.]])
        xt = StandardScaler().fit_transform(x)
        assert np.allclose(xt.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(xt.std(axis=0),  1, atol=1e-10)

    def test_zero_variance_feature(self):
        x = np.array([[1., 5.], [1., 6.], [1., 7.]])
        xt = StandardScaler().fit_transform(x)
        assert np.allclose(xt[:, 0], 0)

    def test_inverse_transform(self):
        x = np.array([[1., 2.], [3., 4.]])
        sc = StandardScaler().fit(x)
        assert np.allclose(sc.inverse_transform(sc.transform(x)), x, atol=1e-10)

    def test_nan_preserved(self):
        x = np.array([[1., np.nan], [3., 4.], [5., 6.]])
        xt = StandardScaler().fit_transform(x)
        assert np.isnan(xt[0, 1])

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            StandardScaler().transform(np.array([[1., 2.]]))

from numcompute.preprocessing import MinMaxScaler

class TestMinMaxScaler:
    def test_range_01(self):
        x = np.array([[0.], [5.], [10.]])
        xt = MinMaxScaler().fit_transform(x)
        assert np.isclose(xt.min(), 0.) and np.isclose(xt.max(), 1.)

    def test_custom_range(self):
        x = np.array([[0.], [10.]])
        xt = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)
        assert np.isclose(xt[0, 0], -1.) and np.isclose(xt[1, 0], 1.)

    def test_constant_feature(self):
        x = np.array([[3., 1.], [3., 2.]])
        xt = MinMaxScaler().fit_transform(x)
        assert np.allclose(xt[:, 0], 0.)

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            MinMaxScaler(feature_range=(5, 1))

from numcompute.preprocessing import SimpleImputer

class TestSimpleImputer:
    def test_mean_imputation(self):
        x = np.array([[1., np.nan], [3., 4.]])
        xt = SimpleImputer(strategy="mean").fit_transform(x)
        assert np.isclose(xt[0, 1], 4.0)

    def test_constant_imputation(self):
        x = np.array([[np.nan, 2.], [3., np.nan]])
        xt = SimpleImputer(strategy="constant", fill_value=-999).fit_transform(x)
        assert np.isclose(xt[0, 0], -999.) and np.isclose(xt[1, 1], -999.)

    def test_median_imputation(self):
        x = np.array([[1.], [3.], [np.nan], [5.]])
        xt = SimpleImputer(strategy="median").fit_transform(x)
        assert np.isclose(xt[2, 0], np.median([1, 3, 5]))

    def test_all_nan_column_stays_nan(self):
        x = np.array([[np.nan], [np.nan]])
        xt = SimpleImputer(strategy="mean").fit_transform(x)
        assert np.all(np.isnan(xt))

from numcompute.preprocessing import OneHotEncoder

class TestOneHotEncoder:
    def test_basic_encoding(self):
        x = np.array([[0.], [1.], [2.], [0.]])
        xt = OneHotEncoder().fit_transform(x)
        assert xt.shape == (4, 3)
        assert np.allclose(xt.sum(axis=1), 1)

    def test_multi_column(self):
        x = np.array([[0., 1.], [1., 0.]])
        xt = OneHotEncoder().fit_transform(x)
        assert xt.shape == (2, 4)

    def test_unknown_category_all_zeros(self):
        ohe = OneHotEncoder().fit(np.array([[0.], [1.], [2.]]))
        xt = ohe.transform(np.array([[99.]]))
        assert np.all(xt == 0)


class TestSort:
    def test_argsort_stable_preserves_tie_order(self):
        arr = np.array([3., 1., 2., 1., 3.])
        idx = argsort_stable(arr)
        sorted_vals = arr[idx]
        assert np.all(sorted_vals[:-1] <= sorted_vals[1:])
        ones = idx[sorted_vals == 1.0]
        assert list(ones) == sorted(ones.tolist())


from numcompute.sort_search import argsort_stable, multi_key_sort

def test_multi_key_sort(self):
        data = np.array([[2,1],[1,3],[1,2]])
        out = multi_key_sort(data, keys=[0,1])
        assert out[0,0]==1 and out[1,0]==1 and out[0,1]==2

def test_multi_key_invalid_key_raises(self):
        with pytest.raises(ValueError):
            multi_key_sort(np.ones((3,2)), keys=[5])

from numcompute.sort_search import topk

class TestTopK:
    def test_top3_largest(self):
        vals = np.array([5.,1.,3.,4.,2.])
        top_vals, _ = topk(vals, 3, largest=True)
        assert set(top_vals) == {3.,4.,5.}

    def test_top3_smallest(self):
        top_vals, _ = topk(np.array([5.,1.,3.,4.,2.]), 3, largest=False)
        assert set(top_vals) == {1.,2.,3.}

    def test_k_exceeds_n_clamped(self):
        top_vals, _ = topk(np.array([1.,2.]), 10)
        assert len(top_vals) == 2

    def test_empty_array(self):
        vals, idx = topk(np.array([]), k=3)
        assert len(vals) == 0 and len(idx) == 0

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            topk(np.ones((3,3)), 2)

from numcompute.sort_search import quickselect

class TestQuickselect:
    def test_largest_1st(self):
        assert quickselect(np.array([3.,1.,4.,1.,5.]), k=1, largest=True) == 5.

    def test_smallest_2nd(self):
        assert quickselect(np.array([3.,1.,4.,1.,5.]), k=2, largest=False) == 1.

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            quickselect(np.array([1.,2.]), k=5)

from numcompute.sort_search import binary_search

class TestBinarySearch:
    def test_found(self):
        idx, found = binary_search(np.array([1,3,5,7,9]), 5)
        assert found and idx == 2

    def test_not_found(self):
        _, found = binary_search(np.array([1,3,5,7,9]), 4)
        assert not found

    def test_insert_at_end(self):
        idx, found = binary_search(np.array([1,2,3]), 99)
        assert idx == 3 and not found

    def test_empty_array(self):
        idx, found = binary_search(np.array([]), 1.)
        assert not found and idx == 0

from numcompute.rank import rank

class TestRank:
    def test_average_ties(self):
        r = rank(np.array([1.,2.,2.,3.]), method="average")
        assert np.isclose(r[1], 2.5) and np.isclose(r[2], 2.5)

    def test_dense_ties(self):
        r = rank(np.array([1.,2.,2.,3.]), method="dense")
        assert r[1]==2 and r[2]==2 and r[3]==3

    def test_ordinal_ties(self):
        assert list(rank(np.array([2.,2.,2.]), method="ordinal")) == [1.,2.,3.]

    def test_nan_gets_nan_rank(self):
        assert np.isnan(rank(np.array([1.,np.nan,3.]))[1])
    
    def test_descending(self):
        r = rank(np.array([1.,2.,3.]), ascending=False)
        assert r[2]==1. and r[0]==3.

    def test_empty_array(self):
        assert len(rank(np.array([]))) == 0

    def test_min_max_methods(self):
        r_min = rank(np.array([5.,5.,5.,9.]), method="min")
        r_max = rank(np.array([5.,5.,5.,9.]), method="max")
        assert np.allclose(r_min[:3], 1.) and np.allclose(r_max[:3], 3.)

from numcompute.rank import percentile

class TestPercentile:
    def test_median(self):
        assert percentile(np.array([1.,2.,3.,4.,5.]), 50) == 3.

    def test_lower_interpolation(self):
        assert percentile(np.array([1.,2.,3.,4.]), 50, interpolation="lower") == 2.

    def test_higher_interpolation(self):
        assert percentile(np.array([1.,2.,3.,4.]), 50, interpolation="higher") == 3.

    def test_nan_ignored(self):
        assert np.isclose(percentile(np.array([1.,np.nan,3.]), 50), 2.)
    
    def test_all_nan_returns_nan(self):
        assert np.isnan(percentile(np.array([np.nan,np.nan]), 50))

    def test_q_out_of_range_raises(self):
        with pytest.raises(ValueError):
            percentile(np.array([1.,2.]), 101)

    def test_multi_q_array(self):
        ps = percentile(np.array([1.,2.,3.,4.,5.]), [0,50,100])
        assert isinstance(ps, np.ndarray) and ps[0]==1. and ps[2]==5.

from numcompute.stats import describe, mean, std, median

class TestDescribe:
    def test_keys_present(self):
        d = describe(np.array([1.,2.,3.,4.,5.]))
        for k in ("n","mean","std","var","min","max","median","q25","q75","nan_count"):
            assert k in d

    def test_nan_count(self):
        d = describe(np.array([1.,np.nan,3.]))
        assert d["nan_count"] == 1 and d["n"] == 2

    def test_mean_std_values(self):
        x = np.array([2.,4.,4.,4.,5.,5.,7.,9.])
        d = describe(x)
        assert np.isclose(d["mean"], x.mean()) and np.isclose(d["std"], x.std())

from numcompute.stats import quantiles, histogram

class TestHistogram:
    def test_counts_sum_to_n(self):
        counts, _ = histogram(np.random.default_rng(0).standard_normal(1000), bins=20)
        assert counts.sum() == 1000

    def test_nan_ignored(self):
        counts, _ = histogram(np.array([1.,2.,np.nan,3.]), bins=3)
        assert counts.sum() == 3

    def test_nan_only_empty(self):
        counts, _ = histogram(np.array([np.nan, np.nan]), bins=5)
        assert counts.sum() == 0

class TestQuantiles:
    def test_median_via_quantile(self):
        assert np.isclose(quantiles(np.array([1.,2.,3.,4.,5.]), 0.5), 3.)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            quantiles(np.array([1.,2.]), 1.5)

from numcompute.stats import WelfordStats

class TestWelford:
    def test_batch_matches_numpy(self):
        data = np.random.default_rng(42).standard_normal((500, 4))
        w = WelfordStats(4); w.update(data)
        assert np.allclose(w.mean_,    data.mean(axis=0),  atol=1e-10)
        assert np.allclose(w.variance, data.var(axis=0),   atol=1e-10)

    def test_sequential_equals_batch(self):
        data = np.random.default_rng(7).standard_normal((200, 3))
        w_seq = WelfordStats(3)
        for row in data:
            w_seq.update(row.reshape(1,-1))
        w_bat = WelfordStats(3); w_bat.update(data)
        assert np.allclose(w_seq.mean_,    w_bat.mean_,    atol=1e-10)
        assert np.allclose(w_seq.variance, w_bat.variance, atol=1e-10)

    def test_sample_variance_ddof1(self):
        data = np.array([[2.],[4.],[4.],[4.],[5.],[5.],[7.],[9.]])
        w = WelfordStats(1); w.update(data)
        assert np.isclose(w.sample_variance[0], np.var(data.ravel(), ddof=1))
        
    def test_empty_batch_noop(self):
        w = WelfordStats(2); w.update(np.ones((5,2)))
        n = w.n_; w.update(np.empty((0,2)))
        assert w.n_ == n

    def test_reset(self):
        w = WelfordStats(1); w.update(np.array([[5.]]))
        w.reset()
        assert w.n_ == 0 and np.isnan(w.variance[0])

from numcompute.metrics import accuracy, confusion_matrix

class TestAccuracy:
    def test_perfect(self):
        y = np.array([0,1,2,1])
        assert accuracy(y, y) == 1.0

    def test_zero(self):
        assert accuracy(np.array([0,0]), np.array([1,1])) == 0.0

    def test_empty(self):
        assert accuracy(np.array([]), np.array([])) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            accuracy(np.array([0,1]), np.array([0]))
    
class TestConfusionMatrix:
    def test_shape(self):
        cm, _ = confusion_matrix(np.array([0,1,2,1,0]), np.array([0,2,2,1,0]))
        assert cm.shape == (3,3)

    def test_diagonal_sum(self):
        y = np.array([0,1,2,1,0])
        p = np.array([0,2,2,1,0])
        cm, _ = confusion_matrix(y, p)
        assert cm.diagonal().sum() == 4

    def test_unsorted_labels(self):
        y_true = np.array([2,0,1,2,0])
        y_pred = np.array([2,0,0,1,0])
        cm, labels = confusion_matrix(y_true, y_pred, labels=np.array([2,0,1]))
        assert cm[0,0] == 1   # true=2, pred=2
        assert cm[1,1] == 2   # true=0, pred=0 (twice)

from numcompute.metrics import precision, recall, f1

class TestPrecisionRecallF1:
    def test_binary_precision_recall(self):
        y_true = np.array([1,1,0,0]); y_pred = np.array([1,0,0,0])
        assert np.isclose(precision(y_true, y_pred, average="binary"), 1.)
        assert np.isclose(recall(y_true, y_pred, average="binary"), .5)

    def test_f1_harmonic_mean(self):
        y_true = np.array([1,1,0,0]); y_pred = np.array([1,0,0,0])
        assert np.isclose(f1(y_true,y_pred,average="binary"), 2*1.*.5/(1.+.5))

    def test_invalid_average_raises(self):
        with pytest.raises(ValueError):
            precision(np.array([0,1]), np.array([0,1]), average="weighted")

    def test_macro_average(self):
        y_true = np.array([0,0,1,1,1]); y_pred = np.array([0,1,1,1,0])
        p = precision(y_true, y_pred, average="macro")
        assert np.isclose(p, (0.5 + 2/3)/2, atol=1e-6)

from numcompute.metrics import mse, r2_score

class TestRegressionMetrics:
    def test_mse_zero(self):
        y = np.array([1.,2.,3.])
        assert mse(y, y) == 0.

    def test_mse_known(self):
        assert np.isclose(mse(np.array([1.,2.,3.]), np.array([2.,2.,2.])), 2/3)

    def test_r2_perfect(self):
        y = np.array([1.,2.,3.])
        assert np.isclose(r2_score(y, y), 1.)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mse(np.array([1.,2.]), np.array([1.]))

from numcompute.metrics import roc_curve, auc

class TestROC:
    def test_perfect_auc(self):
        fpr,tpr,_ = roc_curve(np.array([0,0,1,1]), np.array([.1,.2,.8,.9]))
        assert np.isclose(auc(fpr,tpr), 1.)

    def test_random_auc_near_half(self):
        rng = np.random.default_rng(7)
        y_true = rng.integers(0,2,200); y_score = rng.random(200)
        fpr,tpr,_ = roc_curve(y_true, y_score)
        assert 0.3 < auc(fpr,tpr) < 0.7

    def test_all_positive_no_crash(self):
        fpr,tpr,_ = roc_curve(np.array([1,1,1,1]), np.array([.9,.8,.7,.6]))
        assert np.isfinite(auc(fpr,tpr))

from numcompute.optim import grad

class TestGrad:
    def test_central_quadratic(self):
        g = grad(lambda x: np.sum(x**2), np.array([1.,2.,3.]))
        assert np.allclose(g, [2.,4.,6.], atol=1e-6)

    def test_forward_gradient(self):
        g = grad(lambda x: x[0]*x[1], np.array([2.,3.]), method="forward")
        assert np.allclose(g, [3.,2.], atol=1e-5)

    def test_cubic_1d(self):
        g = grad(lambda x: x[0]**3, np.array([2.]))
        assert np.isclose(g[0], 12., atol=1e-5)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            grad(lambda x: x[0], np.array([1.]), method="bogus")

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            grad(lambda x: x.sum(), np.ones((2,2)))

from numcompute.optim import jacobian

class TestJacobian:
    def test_linear_jacobian(self):
        a = np.array([[1.,2.],[3.,4.]])
        j = jacobian(lambda x: a@x, np.array([1.,1.]))
        assert np.allclose(j, a, atol=1e-6)

    def test_all_three_methods(self):
        f = lambda x: np.array([x[0]**2, x[1]**2])
        x = np.array([3.,4.])
        for method in ("central","forward","backward"):
            j = jacobian(f, x, method=method)
            assert np.allclose(j, np.diag([6.,8.]), atol=1e-4)

from numcompute.utils import euclidean_distance, manhattan_distance, cosine_similarity, pairwise_euclidean

class TestDistances:
    def test_euclidean(self):
        assert np.isclose(euclidean_distance(np.array([0.,0.]), np.array([3.,4.])), 5.)

    def test_manhattan(self):
        assert np.isclose(manhattan_distance(np.array([0.,0.]), np.array([3.,4.])), 7.)

    def test_cosine_parallel(self):
        assert np.isclose(cosine_similarity(np.array([1.,0.]), np.array([1.,0.])), 1.)
    
    def test_cosine_zero_vector(self):
        assert cosine_similarity(np.zeros(3), np.ones(3)) == 0.

    def test_pairwise_shape(self):
        d = pairwise_euclidean(np.random.default_rng(0).standard_normal((5,3)))
        assert d.shape == (5,5) and np.allclose(np.diagonal(d), 0.)

    def test_pairwise_non_square(self):
        d = pairwise_euclidean(
            np.random.default_rng(0).standard_normal((4,3)),
            np.random.default_rng(1).standard_normal((6,3)))
        assert d.shape == (4,6)

    def test_pairwise_symmetry(self):
        x = np.random.default_rng(1).standard_normal((4,2))
        d = pairwise_euclidean(x)
        assert np.allclose(d, d.T, atol=1e-10)
from numcompute.utils import sigmoid, softmax, relu, leaky_relu, logsumexp

class TestActivations:
    def test_sigmoid_midpoint(self):
        assert np.isclose(sigmoid(np.array([0.]))[0], 0.5)

    def test_sigmoid_large_positive_stable(self):
        assert np.isclose(sigmoid(np.array([1000.]))[0], 1.)

    def test_sigmoid_large_negative_stable(self):
        assert np.isclose(sigmoid(np.array([-1000.]))[0], 0.)

    def test_softmax_sums_to_one(self):
        assert np.isclose(softmax(np.array([1.,2.,3.])).sum(), 1.)

    def test_softmax_overflow_stable(self):
        s = softmax(np.array([1000.,1001.,1002.]))
        assert np.all(np.isfinite(s)) and np.isclose(s.sum(), 1.)

    def test_relu(self):
        out = relu(np.array([-2., 0., 3.]))
        assert np.allclose(out, [0., 0., 3.])

    def test_leaky_relu(self):
        out = leaky_relu(np.array([-2., 0., 3.]), alpha=0.1)
        assert np.isclose(out[0], -0.2) and out[1]==0. and out[2]==3.

class TestLogsumexp:
    def test_correctness(self):
        import math
        x = np.array([1.,2.,3.])
        expected = math.log(math.exp(1)+math.exp(2)+math.exp(3))
        assert np.isclose(logsumexp(x), expected)

    def test_overflow_stable(self):
        assert np.isfinite(logsumexp(np.array([1000.,1000.])))
from numcompute.utils import batch_iter

class TestBatchIter:
    def test_sizes(self):
        x = np.arange(10).reshape(-1,1).astype(float)
        sizes = [len(b) for b in batch_iter(x, batch_size=3)]
        assert sizes == [3,3,3,1]

    def test_shuffle_reproducible(self):
        x = np.arange(20).reshape(10,2).astype(float)
        b1 = [b.copy() for b in batch_iter(x, 3, shuffle=True, seed=42)]
        b2 = [b.copy() for b in batch_iter(x, 3, shuffle=True, seed=42)]
        assert all(np.array_equal(a,b) for a,b in zip(b1,b2))

    def test_yields_y_when_provided(self):
        x = np.ones((6,2)); y = np.arange(6)
        batches = list(batch_iter(x, 3, y=y))
        assert len(batches) == 2
        xb, yb = batches[0]
        assert len(xb)==3 and len(yb)==3
from numcompute.pipeline import Transformer, Estimator

class TestProtocols:
    def test_transformer_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Transformer().fit(np.ones((3,2)))

    def test_estimator_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Estimator().fit(np.ones((3,2)), np.ones(3))