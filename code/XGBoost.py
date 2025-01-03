import numpy as np
from XGConfig import Node, XGConfig


class XGBoost:
    """
    XGBoost class for the Gradient Boosting model.
    """

    def __init__(self, config: XGConfig):
        self.root = None
        self.trees = []  # Trained decision trees

        self.config = config
        self.learning_rate = config.learning_rate
        self.n_estimators = config.n_estimators
        self.max_depth = config.max_depth

    def random_prediction(self, data: np.ndarray):
        """
        Initialize the predictions with 0.5 for all data points.
        """
        return np.full(data.shape[0], 0.5)

    def update_predictions(self, predictions: np.ndarray, tree: Node, data: np.ndarray):
        """
        Update the predictions using the new decision tree.
        """
        return (predictions + self.learning_rate * self.predict(tree, data)) / (1 + self.learning_rate)

    def compute_gradients_and_hessians(
        self, predictions: np.ndarray, targets: np.ndarray
    ):
        """
        Compute the gradients and hessians for the log-loss function.
        """
        gradients = predictions - targets  # Gradient for log-loss
        hessians = predictions * (1 - predictions)  # Hessian for log-loss
        return gradients, hessians

    def predict_single(self, tree: Node, row: np.ndarray):
        """
        Predict the value for a single data point.
        """
        while not tree.is_leaf():
            if row[tree.feature] <= tree.threshold:
                tree = tree.left
            else:
                tree = tree.right

        # Return the mean of the targets if not empty else 0.5
        return np.mean(tree.targets) if tree.targets.size > 0 else 0.5

    def predict(self, tree: Node, data: np.ndarray):
        """
        Predict the values for a batch of data points.
        """
        return np.array([self.predict_single(tree, row) for row in data])

    def find_best_split_algo_1(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
    ):
        """
        Find the best split in the decision tree using Algorithm 1.
        """
        best_score = -float("inf")
        best_split = None
        best_feature = None

        num_samples, num_features = data.shape

        for feature in range(num_features):
            sorted_indices = data[:, feature].argsort()
            sorted_data = data[sorted_indices]
            sorted_targets = targets[sorted_indices]
            sorted_gradients = gradients[sorted_indices]
            sorted_hessians = hessians[sorted_indices]

            g_left, h_left = 0, 0
            g_right, h_right = np.sum(sorted_gradients), np.sum(sorted_hessians)

            for i in range(1, num_samples):
                g_left += sorted_gradients[i - 1]
                h_left += sorted_hessians[i - 1]
                g_right -= sorted_gradients[i - 1]
                h_right -= sorted_hessians[i - 1]

                # Compute the score Eq. 7
                score = (
                    (g_left**2) / (h_left + 1e-6)
                    + (g_right**2) / (h_right + self.config.reg_lambda)
                    - (g_left + g_right) ** 2
                    / (h_left + h_right + self.config.reg_lambda)
                ) / 2 - self.config.gamma

                if score > best_score:
                    best_score = score
                    best_split = sorted_data[i - 1, feature]
                    best_feature = feature

        return best_feature, best_split

    def build_tree_algo_1(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        max_depth: int,
    ):
        """
        Recursively build the decision tree using the splitting algorithm 1.
        """
        if max_depth == 1 or data.shape[0] == 0:
            return Node(data, targets, gradients=gradients, hessians=hessians)

        # If self.max_depth is set to 0, then nodes are expanded until all leaves are pure
        if self.max_depth == 0:
            # Check if all targets are the same
            if np.unique(targets).size == 1:
                return Node(data, targets, gradients=gradients, hessians=hessians)

        feature, threshold = self.find_best_split_algo_1(
            data, targets, gradients, hessians
        )

        if feature is None:
            return Node(
                data, targets, gradients=gradients, hessians=hessians
            )  # No split found

        # Split the data
        left_mask = data[:, feature] <= threshold
        right_mask = ~left_mask

        left_data = data[left_mask]
        right_data = data[right_mask]
        left_targets = targets[left_mask]
        right_targets = targets[right_mask]
        left_gradients = gradients[left_mask]
        right_gradients = gradients[right_mask]
        left_hessians = hessians[left_mask]
        right_hessians = hessians[right_mask]

        node = Node(
            data,
            targets,
            gradients=gradients,
            hessians=hessians,
            feature=feature,
            threshold=threshold,
        )

        if (
            left_data.size >= self.config.min_child_weight
            and right_data.size >= self.config.min_child_weight
        ):
            node.left = self.build_tree_algo_1(
                left_data, left_targets, left_gradients, left_hessians, max_depth - 1
            )
            node.right = self.build_tree_algo_1(
                right_data,
                right_targets,
                right_gradients,
                right_hessians,
                max_depth - 1,
            )

        return node

    def fit(self, data: np.ndarray, targets: np.ndarray):
        """
        Fit the model to the data.
        """
        preds = self.random_prediction(data)

        for _ in range(self.n_estimators):

            # Subsample the data if needed
            if self.config.subsample < 1:
                indices = np.random.choice(
                    data.shape[0],
                    size=int(self.config.subsample * data.shape[0]),
                    replace=False,
                )
            else:
                indices = np.arange(data.shape[0])

            current_data = data[indices]
            current_targets = targets[indices]
            current_preds = preds[indices]

            gradients, hessians = self.compute_gradients_and_hessians(
                current_preds, current_targets
            )

            # Randomly select features if needed
            if self.config.colsample_bytree < 1:
                num_features = int(self.config.colsample_bytree * data.shape[1])
                features = np.random.choice(
                    data.shape[1], size=num_features, replace=False
                )
                current_data = current_data[:, features]

            tree = self.build_tree_algo_1(
                current_data, current_targets, gradients, hessians, self.max_depth
            )
            self.trees.append(tree)
            preds[indices] = self.update_predictions(current_preds, tree, current_data)

    def predict_new_data(self, data: np.ndarray):
        """
        Predict the values for new data.
        """
        predictions = np.full(data.shape[0], 0.5)
        for tree in self.trees:
            predictions = (predictions + self.learning_rate * self.predict(tree, data)) / (
                1 + self.learning_rate
            )
        predictions = [1 if p > 0.5 else 0 for p in predictions]
        return predictions

    def score(self, data: np.ndarray, targets: np.ndarray):
        """
        Compute the accuracy of the model.
        """
        predictions = self.predict_new_data(data)
        return np.mean(predictions == targets)


if __name__ == "__main__":
    # Génération des données d'entraînement
    np.random.seed(42)  # Fixer la graine pour des résultats reproductibles

    data_train = np.random.rand(1000, 5)  # Données de taille 20x5
    targets_train = np.random.randint(0, 2, 1000)  # Cibles binaires (0 ou 1)

    # Génération des données de test
    data_test = np.random.rand(10, 5)  # Données de taille 10x5
    targets_test = data_test[:, 0] > 0.5  # Cibles binaires (0 ou 1)

    # Création et entraînement du modèle
    config = XGConfig(
        max_depth=3,
        n_estimators=801,
        min_child_weight=3,
        subsample=0.642807,
        colsample_bytree=0.658399,
    )
    model = XGBoost(config)
    model.fit(data_train, targets_train)

    print([tree.get_height() for tree in model.trees])

    # Calcul de la précision
    accuracy = model.score(data_test, targets_test)
    print(f"Accuracy={accuracy:.4f}")
