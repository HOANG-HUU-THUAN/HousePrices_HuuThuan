from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import joblib
import numpy as np


def train_model(X_train, y_train, model_type="linear_regression"):
    """Huấn luyện mô hình với tinh chỉnh siêu tham số (Hyperparameter Tuning)."""

    if model_type == "linear_regression":
        model = LinearRegression()
        param_grid = {}  # Linear Regression thường không có nhiều hyperparams để tune trong sklearn cơ bản
    elif model_type == "random_forest":
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        }
    else:
        raise ValueError("Loại mô hình không được hỗ trợ")

    # Sử dụng GridSearch để tìm tham số tốt nhất
    # Scoring cho regression thường là r2 hoặc neg_mean_squared_error
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    """Đánh giá mô hình và trả về các chỉ số."""
    y_pred = model.predict(X_test)

    metrics = {
        "r2_score": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    return metrics


def save_model(model, filepath):
    """Lưu mô hình đã huấn luyện vào đĩa."""
    joblib.dump(model, filepath)


def load_model(filepath):
    """Tải mô hình từ đĩa."""
    return joblib.load(filepath)
