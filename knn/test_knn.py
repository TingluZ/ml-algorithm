# pytest -v

import numpy as np
import pytest
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from knn import knn_classifier


@pytest.fixture
def iris_split():
    iris = datasets.load_iris()
    return train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)


@pytest.fixture
def toy_data():
    # two well-separated clusters
    X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_euclidean_known_values():
    model = knn_classifier(n_neighbors=1)
    X1 = np.array([[0.0, 0.0], [3.0, 4.0]])
    X2 = np.array([0.0, 0.0])
    assert np.allclose(model._euclidean(X1, X2), [0.0, 5.0])


def test_get_neighbors_returns_nearest_labels(toy_data):
    X, y = toy_data
    model = knn_classifier(n_neighbors=2)
    model.fit(X, y)
    idx = model._get_neighbors_labels(np.array([0.2, 0.0]))
    assert set(idx) == {0, 0}


def test_predict_toy_data(toy_data):
    X, y = toy_data
    model = knn_classifier(n_neighbors=3)
    model.fit(X, y)
    X_new = np.array([[0.2, 0.0], [10.5, 10.0]])
    assert np.array_equal(model.predict(X_new), [0, 1])


def test_predict_output_shape_and_dtype(iris_split):
    X_train, X_test, y_train, _ = iris_split
    model = knn_classifier(n_neighbors=3)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape == (len(X_test),)
    assert np.issubdtype(pred.dtype, np.integer)


def test_k1_perfect_on_train_set(iris_split):
    X_train, _, y_train, _ = iris_split
    model = knn_classifier(n_neighbors=1)
    model.fit(X_train, y_train)
    assert np.array_equal(model.predict(X_train), y_train)


@pytest.mark.parametrize("k", [1, 3, 5, 7])
def test_matches_sklearn(iris_split, k):
    X_train, X_test, y_train, _ = iris_split
    mine = knn_classifier(n_neighbors=k)
    mine.fit(X_train, y_train)
    ref = KNeighborsClassifier(n_neighbors=k)
    ref.fit(X_train, y_train)
    assert np.array_equal(mine.predict(X_test), ref.predict(X_test))


def test_fit_rejects_mismatched_lengths():
    model = knn_classifier(n_neighbors=1)
    with pytest.raises(AssertionError):
        model.fit(np.zeros((5, 2)), np.zeros(4))


def test_fit_rejects_k_larger_than_samples():
    model = knn_classifier(n_neighbors=10)
    with pytest.raises(AssertionError):
        model.fit(np.zeros((5, 2)), np.zeros(5))
