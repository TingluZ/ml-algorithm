import numpy as np

class knn_classifier():
    def __init__(self, 
                 n_neighbors,
                 metric="euclidean",
                 weights=None):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def _euclidean(self, X1, X2):
        """
        X1: np.ndarray (n_samples, n_features)
        X2: np.ndarray (n_features,)
        """
        distance = np.sqrt(np.sum((X1 - X2) ** 2, axis=-1))
        return distance

    def _manhattan(self):
        pass

    def _get_neighbors(self, x_test):

        if self.metric == "euclidean":
            distance = self._euclidean(self.X_train, x_test)

        distance_sort = np.argsort(distance)
        neighbors = (distance_sort < self.n_neighbors).astype(int)

        return neighbors

    def fit(self, X, y):
        """
        X: np.ndarray (n_samples, n_features)
        y: np.ndarray (n_samples,)
        """

        assert len(X) >= self.n_neighbors, "Sample number should be larger than n_neighbor"
        assert len(X) == len(y), "Train set X and y dimension misalign"
        
        self.X_train = X
        self.y_train = y

    def _pred_one(self, x_test):
        neighbors = self._get_neighbors(x_test)

        return neighbors

    def predict(self, X_test):
        """
        X_test shape (n_test, n_features)
        """
        prediction = np.array()
        for i in range(len(X_test)):
            neighbors = self._pred_one(X_test[i])
            y = self.y_train * neighbors
            y_pred = np.argmax(np.unique(y))
            prediction[i] = y_pred

        return prediction