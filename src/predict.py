import pandas as pd
import joblib


def predict_new_data(model, preprocessor, new_data_df):
    """
    Tiền xử lý dữ liệu mới và thực hiện dự đoán.

    Args:
        model: Đối tượng mô hình đã huấn luyện.
        preprocessor: Đối tượng tiền xử lý đã khớp (fitted).
        new_data_df: DataFrame chứa dữ liệu mới (dạng thô).

    Returns:
        Kết quả dự đoán (giá nhà).
    """
    # Làm sạch cơ bản (đảm bảo kiểu dữ liệu khớp)
    # Ví dụ: nếu có cột nào cần ép kiểu số
    # if "TotalCharges" in new_data_df.columns:
    #     new_data_df["TotalCharges"] = pd.to_numeric(
    #         new_data_df["TotalCharges"], errors="coerce"
    #     )
    #     new_data_df["TotalCharges"].fillna(
    #         0, inplace=True
    #     )

    # Chuyển đổi dữ liệu bằng bộ tiền xử lý đã có
    X_new = preprocessor.transform(new_data_df)

    # Dự đoán
    predictions = model.predict(X_new)

    return predictions
