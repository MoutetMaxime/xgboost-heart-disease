import numpy as np

class Node:
    def __init__(self, data, targets, gradients=None, hessians=None, feature=None, threshold=None):
        self.data = data  # Données à ce nœud
        self.targets = targets  # Cibles à ce nœud
        self.gradients = gradients  # Vecteur des gradients
        self.hessians = hessians  # Vecteur des hessians
        self.feature = feature  # Caractéristique choisie pour le split
        self.threshold = threshold  # Seuil du split
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None

class XGBoost:
    def __init__(self, learning_rate=0.1,max_depth=3,number_trees=100):
        self.root = None
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.number_trees = number_trees
        self.trees = []  # Liste des arbres appris
        self.predictions = None  # Prédictions initiales du modèle

    def init_predictions(self, data):
        """Initialisation des prédictions à une valeur constante (par exemple 0.5)."""
        self.predictions = np.full(data.shape[0], 0.5)
    
    def update_predictions(self, tree, data):
        """Mise à jour des prédictions avec les nouveaux arbres."""
        self.predictions = (self.predictions + self.learning_rate * self.predict(tree, data)) / (1+self.learning_rate)

    def compute_gradients_and_hessians(self, data, targets):
        """Calcul des gradients et des hessians à partir des prédictions actuelles."""
        gradients = self.predictions - targets  # Gradient
        hessians = self.predictions * (1 - self.predictions)  # Hessian pour log-loss
        return gradients, hessians

    def predict_single(self, node, row):
        """Prédit la valeur pour une seule ligne de données."""
        while not node.is_leaf():
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        # Retourner la moyenne des cibles (target) de la feuille
        return np.mean(node.targets) if node.targets.size > 0 else 0.5  # Valeur par défaut si vide

    def predict(self, node, data):
        """Prédit les valeurs pour un ensemble de données."""
        return np.array([self.predict_single(node, row) for row in data])

    def find_best_split_algo_1(self, data, targets, gradients, hessians):
        """Trouver le meilleur split en utilisant les gradients et hessians."""
        best_score = -float('inf')
        best_split = None
        best_feature = None
        
        num_samples, num_features = data.shape

        for feature in range(num_features):  # Boucle sur chaque caractéristique
            sorted_indices = data[:, feature].argsort()
            sorted_data = data[sorted_indices]
            sorted_targets = targets[sorted_indices]
            sorted_gradients = gradients[sorted_indices]
            sorted_hessians = hessians[sorted_indices]

            g_left, h_left = 0, 0
            g_right, h_right = np.sum(sorted_gradients), np.sum(sorted_hessians)

            for i in range(1, num_samples):
                g_left += sorted_gradients[i-1]
                h_left += sorted_hessians[i-1]
                g_right -= sorted_gradients[i-1]
                h_right -= sorted_hessians[i-1]

                # Calcul du score
                score = (g_left**2) / (h_left + 1e-6) + (g_right**2) / (h_right + 1e-6)

                if score > best_score:
                    best_score = score
                    best_split = sorted_data[i-1, feature]
                    best_feature = feature
        
        return best_feature, best_split

    def build_tree_algo_1(self, data, targets, gradients, hessians, max_depth):
        """Construit un arbre récursivement."""
        if max_depth == 0 or data.shape[0] == 0:
            return Node(data, targets, gradients=gradients, hessians=hessians)  # Inclure les gradients et hessians
        
        feature, threshold = self.find_best_split_algo_1(data, targets, gradients, hessians)
        
        if feature is None:
            return Node(data, targets, gradients=gradients, hessians=hessians)  # Aucun split trouvé

        # Séparer les données selon le split
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

        node = Node(data, targets, gradients=gradients, hessians=hessians, feature=feature, threshold=threshold)
        node.left = self.build_tree_algo_1(left_data, left_targets, left_gradients, left_hessians, max_depth - 1)
        node.right = self.build_tree_algo_1(right_data, right_targets, right_gradients, right_hessians, max_depth - 1)

        return node

    def fit(self, data, targets):
        """Apprentissage du modèle."""
        self.init_predictions(data)
        for _ in range(self.number_trees):
            gradients, hessians = self.compute_gradients_and_hessians(data, targets)
            tree = self.build_tree_algo_1(data, targets, gradients, hessians, self.max_depth)
            self.trees.append(tree)
            self.update_predictions(tree, data)
    
    def predict_new_data(self, data):
        """Prédiction des valeurs pour un ensemble de données."""
        predictions = np.full(data.shape[0], 0.5)
        for tree in self.trees:
            predictions = (predictions + self.learning_rate * self.predict(tree, data)) / (1+self.learning_rate)
        predictions = [1 if p > 0.5 else 0 for p in predictions]
        return predictions

# Génération des données d'entraînement
np.random.seed(42)  # Fixer la graine pour des résultats reproductibles
data_train = np.random.rand(20, 5)  # Données de taille 20x5
targets_train = np.random.randint(0, 2, 20)  # Cibles binaires (0 ou 1)

# Génération des données de test
data_test = np.random.rand(10, 5)  # Données de taille 10x5

# Création et entraînement du modèle
model = XGBoost(learning_rate=0.1, max_depth=3, number_trees=10)
model.fit(data_train, targets_train)

# Prédiction sur les données de test
predictions = model.predict_new_data(data_test)
print("Prédictions :", predictions)
