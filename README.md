# ♟️ Phân Loại và Phân Đoạn Quân Cờ Vua bằng ResNet50 & Mask R-CNN

## 🌐 Giới Thiệu

Dự án này là một ứng dụng web được xây dựng bằng **Streamlit**, cho phép người dùng tải lên hình ảnh bàn cờ hoặc các quân cờ riêng lẻ để thực hiện hai tác vụ AI chính:

1.  **Phân loại (Classification):** Dự đoán loại quân cờ chính trong ảnh (Vua, Hậu, Xe, Tượng, Mã, Tốt) bằng mô hình **ResNet50**.
2.  **Phân đoạn (Segmentation):** Phát hiện vị trí các quân cờ và hiển thị kết quả dưới dạng **ảnh nhị phân** bằng **Mask R-CNN**.

## ✨ Tính Năng Chính

* 👉 Giao diện web Streamlit dễ dàng sử dụng.
* 🖼️ Tải lên hình ảnh quân cờ (JPG, JPEG, PNG).
* 📊 Hiển thị kết quả phân loại với độ tin cậy (%).
* 🎨 Hiển thị kết quả phân đoạn dưới dạng mask nhị phân (trắng/đen).
* 🌿 Điều chỉnh ngưỡng tin cậy cho việc phát hiện quân cờ.

## 🧠 Mô Hình Sử Dụng

* **Phân loại:** ResNet50 fine-tuned.
* **Phân đoạn:** Mask R-CNN (ResNet50-FPN) fine-tuned.
  * *Lưu ý:* Mô hình hiện tại chỉ phân biệt **quân cờ** và **nền**, chưa phân biệt từng loại quân.

## ⚙️ Cài Đặt

```bash
git clone https://github.com/blanatole/chess_pieces_classification.git
cd chess_pieces_classification
python -m venv venv
source venv/bin/activate #Linux / MacOS
./venv/Scripts/activate #Windown
pip install -r requirements.txt
```

Tạo thư mục `weight/` và sao chép file trọng số:

```bash
./weight/chess_classifier_resnet_finetuned.pth
./weight/chess_segmentation_rcnn_finetuned.pth
```

## 📺 Cách Sử Dụng

Chạy ứng dụng:

```bash
streamlit run app.py
```

* Tải lên hình ảnh các quân cờ.
* Điều chỉnh ngưỡng tin cậy.
* Xem kết quả phân loại và phân đoạn.

## 📂 Cấu Trúc Thư Mục

```
├── weight/                     # Thư mục chứa file trọng số
├── app.py                # Mã nguồn Streamlit
├── requirements.txt            # Danh sách thư viện cần thiết
└── README.md                   # Hướng dẫn
```

## ⚠️ Hạn Chế

* **Phân loại:** Dự đoán dựa trên toàn bộ ảnh, có thể không chính xác nếu ảnh chứa nhiều quân.
* **Phân đoạn:** Chưa phân biệt loại quân cờ.
* **Hiệu năng:** Phân đoạn chạy trên CPU, chậm hơn GPU.

## 🚀 Hướng Phát Triển

* 💡 Huấn luyện Mask R-CNN với đủ 7 lớp (6 loại quân + nền).
* 💡 Cải thiện accuracy bằng augmentation.
* 💡 Triển khai GPU cho phân đoạn.
* 💡 Nhận diện và diễn giải trạng thái bàn cờ (FEN).

---

## 👥 Nhóm phát triển
- **Nguyễn Minh Ý** (Owner)  
- **Huỳnh Lý Tân Khoa**  
- **Nguyễn Thị Phương Anh**  
- **Võ Thị Như Ý**  
- **Huỳnh Phúc Bảo**

📧 Email liên hệ: `nguyenminhy7714@gmail.com`