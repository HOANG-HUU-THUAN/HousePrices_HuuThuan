import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(filepath):
    """Đọc dữ liệu từ file CSV."""
    df = pd.read_csv(filepath)
    return df


def get_preprocessor(numeric_features, categorical_features):
    """Tạo pipeline tiền xử lý dữ liệu."""

    # Pipeline cho biến số: điền giá trị thiếu và chuẩn hóa
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline cho biến phân loại: điền giá trị thiếu và mã hóa One-Hot
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Kết hợp cả hai pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def preprocess_data(
    df, target_col="median_house_value", test_size=0.2, random_state=42
):
    """Chia dữ liệu và khớp (fit) bộ tiền xử lý."""

    # Xử lý ngoại lai cơ bản (nếu cần, có thể thêm vào đây hoặc thực hiện trước khi gọi hàm này)
    # Trong notebook mẫu có bước loại bỏ ngoại lai dựa trên Z-score,
    # nhưng để giữ cho pipeline đơn giản và giống ProjectGroup11, ta sẽ giữ nguyên dữ liệu hoặc xử lý tối thiểu.
    # Tuy nhiên, nếu muốn giống notebook, ta có thể thêm logic đó vào đây hoặc tách riêng.
    # Ở đây ta sẽ tập trung vào việc chia và transform.

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Xác định loại đặc trưng
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Chia tập train/test
    # Regression không cần stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Lấy bộ tiền xử lý
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    # Khớp và chuyển đổi dữ liệu
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Lấy tên đặc trưng sau khi mã hóa One-Hot
    try:
        onehot_columns = preprocessor.named_transformers_["cat"][
            "onehot"
        ].get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(onehot_columns)
    except:
        feature_names = None  # Dự phòng nếu lỗi

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        feature_names,
    )
