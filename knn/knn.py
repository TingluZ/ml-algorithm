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

    def _get_neighbors_labels(self, x_test):
        """
        Return the labels of k nearst neighbors in the trainset
        """

        if self.metric == "euclidean":
            distance = self._euclidean(self.X_train, x_test)

        distance_sort_idx = np.argsort(distance)
        
        y_train_sorted = self.y_train[distance_sort_idx]

        return y_train_sorted[:self.n_neighbors]

    def fit(self, X, y):
        """
        X: np.ndarray (n_samples, n_features)
        y: np.ndarray (n_samples,)
        """

        assert len(X) >= self.n_neighbors, "Sample number should be larger than n_neighbor"
        assert len(X) == len(y), "Train set X and y dimension misalign"
        
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        """
        X_test shape (n_test, n_features)
        """
        prediction = np.zeros(len(X_test))
        for i in range(len(X_test)):
            neighbors = self._get_neighbors_labels(X_test[i])
            y_unique, y_counts = np.unique(neighbors, return_counts=True) # get the unique labels and the counts
            y_pred_idx = np.argmax(y_counts) # get the most common label (maximum counts)
            y_pred = y_unique[y_pred_idx] # the prediction is the label with maximum counts
            prediction[i] = y_pred

        return prediction.astype(int)