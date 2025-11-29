# Dự Đoán Giá Nhà California (California House Price Prediction)

Dự án này sử dụng Machine Learning để dự đoán giá nhà trung bình tại California dựa trên các đặc điểm như vị trí, số phòng, dân số, thu nhập, v.v. Quy trình thực hiện tuân theo chuẩn **CRISP-DM**.

## Cấu Trúc Dự Án

*   `data/`: Chứa dữ liệu (`housing.csv`).
*   `notebooks/`: Chứa Jupyter Notebook (`house-price-prediction.ipynb`) thực hiện quy trình phân tích và huấn luyện mô hình.
*   `src/`: Mã nguồn Python cho các bước xử lý dữ liệu, huấn luyện và dự đoán.
    *   `preprocessing.py`: Tiền xử lý dữ liệu.
    *   `modeling.py`: Huấn luyện và đánh giá mô hình.
    *   `predict.py`: Hàm dự đoán cho dữ liệu mới.
*   `models/`: Chứa mô hình đã huấn luyện (`rf_model.pkl`) và bộ tiền xử lý (`preprocessor.pkl`).
*   `demo/`: Ứng dụng Web Streamlit (`app.py`) để demo.

## Yêu Cầu Hệ Thống

*   Python 3.8 trở lên.
*   Các thư viện Python: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `joblib`.

## Hướng Dẫn Cài Đặt

1.  Clone dự án về máy:
    ```bash
    git clone https://github.com/HOANG-HUU-THUAN/HousePrices_HuuThuan.git
    cd HousePrices_HuuThuan
    ```

2.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```
    *(Nếu chưa có file `requirements.txt`, bạn có thể cài thủ công: `pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib`)*

## Hướng Dẫn Sử Dụng

### 1. Huấn Luyện Mô Hình
Mở và chạy file notebook để thực hiện phân tích dữ liệu và huấn luyện mô hình:
*   Đường dẫn: `notebooks/house-price-prediction.ipynb`
*   Sau khi chạy xong, mô hình và preprocessor sẽ được lưu vào thư mục `models/`.

### 2. Chạy Ứng Dụng Demo
Sử dụng Streamlit để chạy giao diện web dự đoán giá nhà:

```bash
streamlit run demo/app.py
```

Truy cập vào đường dẫn hiển thị trên terminal (thường là `http://localhost:8501`) để sử dụng.

---
**Tác giả:** Hoàng Hữu Thuận
