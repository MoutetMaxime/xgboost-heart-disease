import numpy as np


class XGConfig:
    def __init__(
            self,
            learning_rate: float = 0.3,
            n_estimators: int = 100,
            max_depth: int = 3,
            min_child_weight: int = 1,
            colsample_bytree: float = 1.0,
            subsample: float = 1.0,
            gamma: float = 0.0,
            reg_alpha: float = 0.0,
            reg_lambda: float = 1.0
        ):
        """
            Configuration class for the XGBoost model.

            Args:
                learning_rate (float): The learning rate of the model.
                n_estimators (int): The number of trees in the model.
                max_depth (int): The maximum depth of the trees. If max_depth is set to 0, then nodes are expanded until all leaves are pure.
                min_child_weight (int): The minimum number of samples required to create a new node.
                colsample_bytree (float): The fraction of features to consider when creating a new tree.
                subsample (float): The fraction of samples to consider when creating a new tree.
                gamma (float): The minimum loss reduction required to make a further partition on a leaf node of the tree.
                reg_alpha (float): The L1 regularization term.
                reg_lambda (float): The L2 regularization term.
        """
        assert 0 < learning_rate <= 1, "The learning rate must be between 0 and 1."
        assert n_estimators > 0, "The number of estimators must be greater than 0."
        assert max_depth >= 0, "The maximum depth must be greater than or equal to 0."
        assert min_child_weight >= 0, "The minimum child weight must be greater than or equal to 0."
        assert 0 < colsample_bytree <= 1, "The fraction of features to consider must be between 0 and 1."
        assert 0 < subsample <= 1, "The fraction of samples to consider must be between 0 and 1."
        assert gamma >= 0, "The minimum loss reduction must be greater than or equal to 0."
        assert reg_alpha >= 0, "The L1 regularization term must be greater than or equal to 0."
        assert reg_lambda >= 0, "The L2 regularization term must be greater than or equal to 0."

        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda


class Node:
    """
       Node class for the Decision Tree.
    """
    def __init__(self, data: np.ndarray, targets: np.ndarray, gradients: np.ndarray=None, hessians: np.ndarray=None, feature: str=None, threshold: float=None):
        self.data = data  # Node data
        self.targets = targets  # Node targets
        self.gradients = gradients  # Gradient vector
        self.hessians = hessians  # Hessian vector
        self.feature = feature  # Split feature
        self.threshold = threshold  # Split threshold
        self.left = None # Left child
        self.right = None # Right child

    def is_leaf(self):
        return self.left is None and self.right is None
    
    def get_height(self):
        if self.is_leaf():
            return 1
        return 1 + max(self.left.get_height(), self.right.get_height())
