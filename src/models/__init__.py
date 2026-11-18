"""
Base model interface and 5 model implementations for PathSnap candidate classification.
"""

from abc import ABC, abstractmethod
import numpy as np
import time
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class BaseModel(ABC):
    def __init__(self, name: str, algorithm_family: str):
        self.name = name
        self.algorithm_family = algorithm_family
        self.model = None
        self.scaler = StandardScaler()
        self.training_time_sec = 0.0
        self.training_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        t0 = time.time()
        X_scaled = self.scaler.fit_transform(X)
        self._fit_internal(X_scaled, y)
        self.training_time_sec = time.time() - t0

    @abstractmethod
    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X_scaled)
            prob1 = 1.0 / (1.0 + np.exp(-df))
            return np.column_stack([1.0 - prob1, prob1])
        else:
            preds = self.model.predict(X_scaled)
            return np.column_stack([1.0 - preds, preds])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class PathSnapLogisticRegression(BaseModel):
    def __init__(self, C: float = 1.0, max_iter: int = 500, class_weight: str = "balanced"):
        super().__init__("Logistic Regression", "Linear and probabilistic")
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=42
        )

    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)


class PathSnapDecisionTree(BaseModel):
    def __init__(self, max_depth: int = 12, min_samples_split: int = 20, min_samples_leaf: int = 10):
        super().__init__("Decision Tree", "Tree-based")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=42
        )

    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)


class PathSnapRandomForest(BaseModel):
    def __init__(self, n_estimators: int = 80, max_depth: int = 14, min_samples_leaf: int = 5):
        super().__init__("Random Forest", "Ensemble")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)


class PathSnapGradientBoosting(BaseModel):
    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_depth: int = 8,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0
    ):
        super().__init__("Gradient Boosting", "Boosting")
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            class_weight="balanced",
            random_state=42
        )

    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        if hasattr(self.model, "train_score_"):
            self.training_history = list(self.model.train_score_)


class PathSnapMLP(BaseModel):
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (64, 32), max_iter: int = 150, alpha: float = 0.001):
        super().__init__("Multi-Layer Perceptron", "Neural networks")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.alpha = alpha
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            alpha=alpha,
            early_stopping=True,
            random_state=42
        )

    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        if hasattr(self.model, "loss_curve_"):
            self.training_history = list(self.model.loss_curve_)


def get_all_models() -> Dict[str, BaseModel]:
    return {
        "Logistic_Regression": PathSnapLogisticRegression(),
        "Decision_Tree": PathSnapDecisionTree(),
        "Random_Forest": PathSnapRandomForest(),
        "Gradient_Boosting": PathSnapGradientBoosting(),
        "MLP_Neural_Net": PathSnapMLP()
    }
