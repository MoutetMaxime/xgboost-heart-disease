import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from XGBoost import XGBoost
from XGConfig import XGConfig


def seed_everything(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def load_data():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets["num"]

    y[y != 0] = 1

    return X, y


def encode_categorical_features(X_train, X_test):
    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    # Encodage des colonnes catégorielles
    X_train_encoded = pd.get_dummies(
        X_train, columns=categorical_features, dummy_na=True, drop_first=False
    )
    X_test_encoded = pd.get_dummies(
        X_test, columns=categorical_features, dummy_na=True, drop_first=False
    )

    X_train_encoded = X_train_encoded.loc[
        :, X_train_encoded.apply(pd.Series.nunique) > 1
    ]
    X_test_encoded = X_test_encoded.reindex(
        columns=X_train_encoded.columns, fill_value=0
    )

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
    seed_everything(42)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_encoded, X_test_encoded = encode_categorical_features(X_train, X_test)

    # model = xgb.XGBClassifier(learning_rate=0.309467, n_estimators=801, max_depth=3, min_child_weight=3, colsample_bytree=0.658399, subsample=0.642807, gamma=0.761624, reg_alpha=1.135854, reg_lambda=97.30065)
    config = XGConfig(
        learning_rate=0.309467,
        n_estimators=801,
        max_depth=3,
        min_child_weight=3,
        colsample_bytree=0.658399,
        subsample=0.642807,
        gamma=0.761624,
        reg_alpha=1.135854,
        reg_lambda=97.30065,
    )
    model = XGBoost(config)
    model.fit(X_train_encoded.to_numpy(), y_train.to_numpy())

    baseline = xgb.XGBClassifier(
        learning_rate=0.309467,
        n_estimators=801,
        max_depth=3,
        min_child_weight=3,
        colsample_bytree=0.658399,
        subsample=0.642807,
        gamma=0.761624,
        reg_alpha=1.135854,
        reg_lambda=97.30065,
        random_state=42,
    )
    baseline.fit(X_train_encoded, y_train)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    random_forest = RandomForestClassifier(n_estimators=801, random_state=42)
    random_forest.fit(X_train_encoded, y_train)

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_encoded, y_train)

    # Prédiction
    y_train_pred = model.predict_new_data(X_train_encoded.to_numpy())
    y_test_pred = model.predict_new_data(X_test_encoded.to_numpy())

    y_train_pred_b = baseline.predict(X_train_encoded)
    y_test_pred_b = baseline.predict(X_test_encoded)

    y_train_pred_rf = random_forest.predict(X_train_encoded)
    y_test_pred_rf = random_forest.predict(X_test_encoded)

    y_train_pred_lr = log_reg.predict(X_train_encoded)
    y_test_pred_lr = log_reg.predict(X_test_encoded)

    # Calcul des métriques
    accuracy_train, specificity_train, sensitivity_train, f1_train = get_metrics(
        y_train_pred, y_train
    )
    accuracy_test, specificity_test, sensitivity_test, f1_test = get_metrics(
        y_test_pred, y_test
    )

    accuracy_train_b, specificity_train_b, sensitivity_train_b, f1_train_b = (
        get_metrics(y_train_pred_b, y_train)
    )
    accuracy_test_b, specificity_test_b, sensitivity_test_b, f1_test_b = get_metrics(
        y_test_pred_b, y_test
    )

    accuracy_train_rf, specificity_train_rf, sensitivity_train_rf, f1_train_rf = (
        get_metrics(y_train_pred_rf, y_train)
    )
    accuracy_test_rf, specificity_test_rf, sensitivity_test_rf, f1_test_rf = (
        get_metrics(y_test_pred_rf, y_test)
    )

    accuracy_train_lr, specificity_train_lr, sensitivity_train_lr, f1_train_lr = (
        get_metrics(y_train_pred_lr, y_train)
    )
    accuracy_test_lr, specificity_test_lr, sensitivity_test_lr, f1_test_lr = (
        get_metrics(y_test_pred_lr, y_test)
    )

    print(
        f"Train: Accuracy={accuracy_train:.4f}, Specificity={specificity_train:.4f}, Sensitivity={sensitivity_train:.4f}, F1={f1_train:.4f}"
    )
    print(
        f"Test: Accuracy={accuracy_test:.4f}, Specificity={specificity_test:.4f}, Sensitivity={sensitivity_test:.4f}, F1={f1_test:.4f}"
    )

    print(
        f"Train (baseline): Accuracy={accuracy_train_b:.4f}, Specificity={specificity_train_b:.4f}, Sensitivity={sensitivity_train_b:.4f}, F1={f1_train_b:.4f}"
    )
    print(
        f"Test (baseline): Accuracy={accuracy_test_b:.4f}, Specificity={specificity_test_b:.4f}, Sensitivity={sensitivity_test_b:.4f}, F1={f1_test_b:.4f}"
    )

    print(
        f"Train (random forest): Accuracy={accuracy_train_rf:.4f}, Specificity={specificity_train_rf:.4f}, Sensitivity={sensitivity_train_rf:.4f}, F1={f1_train_rf:.4f}"
    )
    print(
        f"Test (random forest): Accuracy={accuracy_test_rf:.4f}, Specificity={specificity_test_rf:.4f}, Sensitivity={sensitivity_test_rf:.4f}, F1={f1_test_rf:.4f}"
    )

    print(
        f"Train (logistic regression): Accuracy={accuracy_train_lr:.4f}, Specificity={specificity_train_lr:.4f}, Sensitivity={sensitivity_train_lr:.4f}, F1={f1_train_lr:.4f}"
    )
    print(
        f"Test (logistic regression): Accuracy={accuracy_test_lr:.4f}, Specificity={specificity_test_lr:.4f}, Sensitivity={sensitivity_test_lr:.4f}, F1={f1_test_lr:.4f}"
    )

    # Confusion matrix
    print("Confusion matrix (Train):")
    print(confusion_matrix(y_train, y_train_pred))
    print("Confusion matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))

    print("Confusion matrix (Train) (baseline):")
    print(confusion_matrix(y_train, y_train_pred_b))
    print("Confusion matrix (Test) (baseline):")
    print(confusion_matrix(y_test, y_test_pred_b))

    print("Confusion matrix (Train) (random forest):")
    print(confusion_matrix(y_train, y_train_pred_rf))
    print("Confusion matrix (Test) (random forest):")
    print(confusion_matrix(y_test, y_test_pred_rf))

    print("Confusion matrix (Train) (logistic regression):")
    print(confusion_matrix(y_train, y_train_pred_lr))
    print("Confusion matrix (Test) (logistic regression):")
    print(confusion_matrix(y_test, y_test_pred_lr))

    # Save metrics in csv
    metrics = pd.DataFrame(
        {
            "Model": [
                "Custom XGBoost",
                "XGBoost",
                "Random Forest",
                "Logistic Regression",
            ],
            "Accuracy Train": [
                accuracy_train,
                accuracy_train_b,
                accuracy_train_rf,
                accuracy_train_lr,
            ],
            "Specificity Train": [
                specificity_train,
                specificity_train_b,
                specificity_train_rf,
                specificity_train_lr,
            ],
            "Sensitivity Train": [
                sensitivity_train,
                sensitivity_train_b,
                sensitivity_train_rf,
                sensitivity_train_lr,
            ],
            "F1 Train": [f1_train, f1_train_b, f1_train_rf, f1_train_lr],
            "Accuracy Test": [
                accuracy_test,
                accuracy_test_b,
                accuracy_test_rf,
                accuracy_test_lr,
            ],
            "Specificity Test": [
                specificity_test,
                specificity_test_b,
                specificity_test_rf,
                specificity_test_lr,
            ],
            "Sensitivity Test": [
                sensitivity_test,
                sensitivity_test_b,
                sensitivity_test_rf,
                sensitivity_test_lr,
            ],
            "F1 Test": [f1_test, f1_test_b, f1_test_rf, f1_test_lr],
        }
    )

    # metrics.to_csv("metrics.csv", index=False)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    cm_train_b = confusion_matrix(y_train, y_train_pred_b)
    cm_test_b = confusion_matrix(y_test, y_test_pred_b)

    cm_train_rf = confusion_matrix(y_train, y_train_pred_rf)
    cm_test_rf = confusion_matrix(y_test, y_test_pred_rf)

    cm_train_lr = confusion_matrix(y_train, y_train_pred_lr)
    cm_test_lr = confusion_matrix(y_test, y_test_pred_lr)

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Listes pour itérer facilement
    matrices = [cm_test_lr, cm_test_rf, cm_test_b, cm_test]
    titles = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Custom XGBoost",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 6))

    # Normalisation pour l'échelle des couleurs communes
    vmin = 0
    vmax = max(
        np.max(cm_test),
        np.max(cm_test_b),
        np.max(cm_test_rf),
        np.max(cm_test_lr),
    )

    for j, cm in enumerate(matrices):  # j = index du modèle
        ax = axes[j]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.5},
        )

        # Labels et titres uniquement pour la première matrice de chaque ligne
        if j == 0:
            ax.set_ylabel("True Labels")
            ax.set_yticklabels(["0", "1"], rotation=0)
        # Pas de labels pour les autres matrices
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        ax.set_xlabel("Predicted Labels")
        ax.set_xticklabels(["0", "1"], rotation=0)

        ax.set_title(titles[j], fontsize=12, fontweight="bold")

    # Ajustement de l'espacement
    plt.tight_layout()
    plt.show()
