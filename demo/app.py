import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Cấu hình trang (Phải đặt đầu tiên)
st.set_page_config(
    page_title="Dự Đoán Giá Nhà California",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Thêm thư mục src vào đường dẫn hệ thống
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict_new_data

# Đường dẫn đến model và preprocessor
# Đường dẫn đến model và preprocessor (Sử dụng đường dẫn tuyệt đối)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "rf_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "..", "models", "preprocessor.pkl")


@st.cache_resource
def load_artifacts():
    """Tải model và preprocessor, có cache để tăng tốc độ."""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except FileNotFoundError:
        return None, None


model, preprocessor = load_artifacts()

# Tiêu đề và mô tả
st.title("🏠 Hệ Thống Dự Đoán Giá Nhà California")
st.markdown("---")

if model is None:
    st.error(
        "⚠️ Lỗi: Không tìm thấy Model hoặc Preprocessor. Vui lòng kiểm tra lại thư mục models hoặc chạy script huấn luyện."
    )
else:
    # Form nhập liệu chính
    with st.form("house_price_prediction_form"):
        st.subheader("📝 Thông tin ngôi nhà")

        # Chia layout thành 3 cột
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📍 Vị trí")
            longitude = st.number_input("Kinh độ (Longitude)", value=-122.23, step=0.01)
            latitude = st.number_input("Vĩ độ (Latitude)", value=37.88, step=0.01)
            ocean_proximity = st.selectbox(
                "Vị trí gần biển",
                ("NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"),
            )

        with col2:
            st.markdown("### 🏗️ Đặc điểm")
            housing_median_age = st.slider("Tuổi nhà trung bình (Năm)", 1, 100, 30)
            total_rooms = st.number_input(
                "Tổng số phòng", min_value=1, value=1000, step=10
            )
            total_bedrooms = st.number_input(
                "Tổng số phòng ngủ", min_value=1, value=200, step=10
            )

        with col3:
            st.markdown("### 👥 Dân cư & Thu nhập")
            population = st.number_input(
                "Dân số khu vực", min_value=1, value=500, step=10
            )
            households = st.number_input(
                "Số hộ gia đình", min_value=1, value=200, step=10
            )
            median_income = st.number_input(
                "Thu nhập trung bình (x10.000$)", min_value=0.0, value=5.0, step=0.1
            )

        # Nút submit
        submitted = st.form_submit_button(
            "🚀 Dự Đoán Giá", use_container_width=True, type="primary"
        )

    # Xử lý khi nhấn nút
    if submitted:
        # Tạo dataframe từ input
        data = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity,
        }
        input_df = pd.DataFrame(data, index=[0])

        # Dự đoán
        prediction = predict_new_data(model, preprocessor, input_df)
        predicted_price = prediction[0]

        st.markdown("---")
        st.subheader("🎯 Kết quả dự đoán")

        # Hiển thị kết quả
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric(
                label="Giá nhà dự đoán",
                value=f"${predicted_price:,.2f}",
                delta="USD",
            )

        with res_col2:
            st.info(
                f"💰 Với thu nhập trung bình {median_income * 10000:,.0f} USD, giá nhà ước tính là **${predicted_price:,.2f}**."
            )

            # Thêm một số nhận xét đơn giản dựa trên giá
            if predicted_price > 500000:
                st.warning("🔥 Đây là khu vực có giá trị bất động sản rất cao!")
            elif predicted_price < 100000:
                st.success("✨ Đây là khu vực có giá cả phải chăng.")
