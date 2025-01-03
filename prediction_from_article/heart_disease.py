import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from .Tree import XGBoost

def load_data():
    heart_disease = fetch_ucirepo(id=45) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets["num"]

    y[y != 0] = 1

    return X, y


def encode_categorical_features(X_train, X_test):
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


    # Encodage des colonnes catégorielles
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, dummy_na=True, drop_first=False)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, dummy_na=True, drop_first=False)

    X_train_encoded = X_train_encoded.loc[:, X_train_encoded.apply(pd.Series.nunique) > 1]
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    return X_train_encoded, X_test_encoded


def get_metrics(y_pred, y_true):
    cm_train = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm_train.ravel()  # Décompose les valeurs de la matrice

    # Calcul des métriques
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # Rappel (ou sensibilité)
    specificity = tn / (tn + fp)  # Calcul manuel de la spécificité
    f1 = f1_score(y_true, y_pred)

    return accuracy, specificity, sensitivity, f1


if __name__ == "__main__":
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train_encoded, X_test_encoded = encode_categorical_features(X_train, X_test)

    #model = xgb.XGBClassifier(learning_rate=0.309467, n_estimators=801, max_depth=3, min_child_weight=3, colsample_bytree=0.658399, subsample=0.642807, gamma=0.761624, reg_alpha=1.135854, reg_lambda=97.30065)
    model = XGBoost(learning_rate=0.1, max_depth=3, number_trees=10)
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    # Prédiction
    y_train_pred = model.predict_new_data(X_train.to_numpy())
    y_test_pred = model.predict_new_data(X_test.to_numpy())

    # Calcul des métriques
    accuracy_train, specificity_train, sensitivity_train, f1_train = get_metrics(y_train_pred, y_train)
    accuracy_test, specificity_test, sensitivity_test, f1_test = get_metrics(y_test_pred, y_test)

    print(f"Train: Accuracy={accuracy_train:.4f}, Specificity={specificity_train:.4f}, Sensitivity={sensitivity_train:.4f}, F1={f1_train:.4f}")
    print(f"Test: Accuracy={accuracy_test:.4f}, Specificity={specificity_test:.4f}, Sensitivity={sensitivity_test:.4f}, F1={f1_test:.4f}")

    # Confusion matrix
    print("Confusion matrix (Train):")
    print(confusion_matrix(y_train, y_train_pred))
    print("Confusion matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))
