import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms as T
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

# --- Configuration ---
# Đảm bảo đường dẫn này đúng trong môi trường chạy Streamlit
CLASSIFIER_MODEL_PATH = './weight/chess_classifier_resnet_finetuned.pth'
SEGMENTATION_MODEL_PATH = './weight/chess_segmentation_rcnn_finetuned.pth'

# --- Class Names ---
CLASS_NAMES_CLASSIFIER = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']
# Vẫn cần định nghĩa CLASS_NAMES_SEGMENTATION dù không hiển thị label trực tiếp
# vì model trả về index dựa trên số lớp này
CLASS_NAMES_SEGMENTATION = ['background', 'piece']

# --- Device Setup ---
main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Chạy segmentation trên CPU để tránh lỗi NMS trên CUDA (nếu có)
segmentation_device = torch.device("cpu")

# --- Model Loading Functions with Caching ---

@st.cache_resource
def load_classifier_model(model_path):
    """Tải mô hình ResNet50 để phân loại."""
    try:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES_CLASSIFIER))
        try:
            # Ưu tiên load map_location=main_device
            model.load_state_dict(torch.load(model_path, map_location=main_device, weights_only=True))
        except:
            st.warning(f"Không thể load classifier weights với weights_only=True hoặc trên device {main_device}. Thử lại với CPU.")
            try:
                # Nếu lỗi, thử load lên CPU rồi chuyển device sau
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            except:
                 st.warning(f"Không thể load classifier weights với weights_only=True ngay cả trên CPU. Thử không dùng tham số này.")
                 model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        model = model.to(main_device) # Đảm bảo model ở đúng device chính
        model.eval()
        st.success(f"Classifier model loaded successfully on {main_device}!")
        return model
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file trọng số phân loại tại: {model_path}")
        return None
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình phân loại: {e}")
        return None

@st.cache_resource
def load_segmentation_model(model_path, num_classes):
    """Tải mô hình Mask R-CNN để phân đoạn (CHẠY TRÊN CPU)."""
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        try:
             model.load_state_dict(torch.load(model_path, map_location=segmentation_device, weights_only=True))
        except:
             st.warning(f"Không thể load segmentation weights với weights_only=True. Thử lại không có tham số này.")
             model.load_state_dict(torch.load(model_path, map_location=segmentation_device))

        model.to(segmentation_device) # Đảm bảo model nằm trên CPU
        model.eval()
        st.success(f"Segmentation model loaded successfully on {segmentation_device}!")
        return model
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file trọng số phân đoạn tại: {model_path}")
        return None
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình phân đoạn: {e}")
        return None

# --- Preprocessing Functions ---

def preprocess_image_for_classification(image_pil):
    """Chuẩn bị ảnh cho mô hình phân loại ResNet50."""
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image_pil).unsqueeze(0)
    return image_tensor.to(main_device)

def preprocess_image_for_segmentation(image_pil):
    """Chuẩn bị ảnh cho mô hình phân đoạn Mask R-CNN (GỬI LÊN CPU)."""
    transform = T.Compose([
        T.ToTensor()
    ])
    image_tensor = transform(image_pil)
    return [image_tensor.to(segmentation_device)]

# --- Inference Functions ---

def classify_image(model, processed_image):
    """Thực hiện phân loại ảnh bằng model ResNet50."""
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = torch.softmax(outputs, dim=1)
        top_p, top_class_idx = probabilities.topk(1, dim=1)
    return top_class_idx.item(), top_p.item()

def segment_image(model, processed_image_list):
    """Thực hiện phân đoạn ảnh bằng model Mask R-CNN (CHẠY TRÊN CPU)."""
    with torch.no_grad():
        outputs = model(processed_image_list)
    return outputs[0]

# --- Visualization Function (MODIFIED) ---

def create_binary_mask_image(image_pil, outputs, threshold):
    """
    Tạo ảnh nhị phân từ kết quả segmentation.
    Quân cờ (trên ngưỡng) màu trắng (255), nền màu đen (0).
    """
    # Lấy kích thước ảnh gốc để tạo ảnh mask cùng kích thước
    try:
        # Nếu image_pil là PIL Image
        width, height = image_pil.size
    except AttributeError:
        # Nếu image_pil đã là numpy array (ít khả năng xảy ra với code hiện tại)
         height, width, _ = image_pil.shape

    # Tạo ảnh nền đen (grayscale)
    binary_mask_image = np.zeros((height, width), dtype=np.uint8)

    # Lấy scores và masks từ output (đã ở CPU)
    scores = outputs['scores'].numpy()
    masks = outputs['masks'].numpy() # Shape (N, 1, H, W)

    # Duyệt qua các đối tượng được phát hiện
    for i in range(len(scores)):
        score = scores[i]
        # Chỉ xử lý những mask có score cao hơn ngưỡng
        if score > threshold:
            # Lấy mask của đối tượng thứ i, bỏ chiều channel (1), shape (H, W)
            mask = masks[i, 0]
            # Tạo mask boolean (True/False) dựa trên ngưỡng mask (thường là 0.5)
            mask_bool = mask > 0.5
            # Tại những vị trí True trong mask_bool, gán giá trị trắng (255) cho ảnh nhị phân
            # Phép OR (|) logic đảm bảo pixel trắng nếu có bất kỳ mask nào phủ lên
            binary_mask_image[mask_bool] = 255

    # Chuyển numpy array (H, W) thành ảnh PIL Grayscale ('L')
    return Image.fromarray(binary_mask_image, mode='L')

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Chess Piece Recognition")
st.title("♟️ Chess Piece Classification and Segmentation")
st.write("Upload an image of a chessboard or chess pieces.")

# Tải models
classifier_model = load_classifier_model(CLASSIFIER_MODEL_PATH)
segmentation_model = load_segmentation_model(SEGMENTATION_MODEL_PATH, num_classes=len(CLASS_NAMES_SEGMENTATION))

# --- Sidebar ---
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
segmentation_threshold = st.sidebar.slider("Segmentation Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")

    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Classification Result (ResNet50)")
        if classifier_model:
            try:
                processed_cls_img = preprocess_image_for_classification(image_pil)
                pred_idx, confidence = classify_image(classifier_model, processed_cls_img)
                pred_label = CLASS_NAMES_CLASSIFIER[pred_idx]
                st.metric(label="Predicted Piece Type", value=pred_label.capitalize(), delta=f"{confidence*100:.2f}% Confidence")
                st.info("Lưu ý: Phân loại trên toàn bộ ảnh có thể kém chính xác. Mô hình này hoạt động tốt nhất với ảnh chứa một quân cờ.")
            except Exception as e:
                st.error(f"Lỗi trong quá trình phân loại: {e}")
        else:
            st.warning("Classifier model not loaded. Cannot perform classification.")

    with col2:
        # Sửa tiêu đề cột kết quả segmentation
        st.subheader("🎨 Segmentation Result (Binary Mask)")
        if segmentation_model:
            try:
                processed_seg_img = preprocess_image_for_segmentation(image_pil)
                outputs = segment_image(segmentation_model, processed_seg_img)

                # Gọi hàm tạo ảnh nhị phân mới
                result_image = create_binary_mask_image(image_pil, outputs, segmentation_threshold)

                # Hiển thị ảnh nhị phân
                st.image(result_image, caption=f"Binary Mask (Threshold: {segmentation_threshold:.2f})", use_column_width=True)

                # Hiển thị thông tin chi tiết vẫn có thể hữu ích để biết số lượng phát hiện
                with st.expander("Show Detection Details (Scores)"):
                     scores = outputs['scores'].numpy()
                     labels = outputs['labels'].numpy() # Dù không dùng label để vẽ nhưng vẫn lấy được
                     valid_indices = scores > segmentation_threshold
                     # Chỉ hiển thị score của các object trên ngưỡng
                     detected_scores = scores[valid_indices]
                     st.write(f"{len(detected_scores)} objects detected above threshold:")
                     st.write(detected_scores)

            except Exception as e:
                st.error(f"Lỗi trong quá trình phân đoạn: {e}")
        else:
            st.warning("Segmentation model not loaded. Cannot perform segmentation.")

else:
    st.info("Please upload an image using the sidebar to start.")

st.sidebar.markdown("---")
st.sidebar.write("Model Information:")
st.sidebar.write(f"- Classifier: ResNet50 (Classes: {len(CLASS_NAMES_CLASSIFIER)}, Device: {main_device.type.upper()})")
# Cập nhật thông tin device cho segmentation
st.sidebar.write(f"- Segmenter: Mask R-CNN (Classes: {len(CLASS_NAMES_SEGMENTATION)}, Device: {segmentation_device.type.upper()})")