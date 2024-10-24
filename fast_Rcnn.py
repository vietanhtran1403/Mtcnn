import torch
import torchvision
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os

# Bước 1: Phát hiện khuôn mặt từ webcam bằng Faster R-CNN
def detect_faces_from_webcam(model, device):
    model.eval()  # Đặt mô hình ở chế độ đánh giá (eval mode)
    cap = cv2.VideoCapture(0)  # Mở webcam

    if not cap.isOpened():
        print("Unable to access webcam!")
        return

    while True:
        ret, frame = cap.read()  # Đọc khung hình từ webcam
        if not ret:
            print("Unable to read frames from webcam!")
            break

        # Chuyển đổi frame sang tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = ToTensor()(frame_rgb).unsqueeze(0).to(device)  # Thêm batch dimension và chuyển sang GPU

        # Phát hiện khuôn mặt
        with torch.no_grad():
            predictions = model(frame_tensor)  # Đưa frame vào mô hình để dự đoán

        # Lặp qua các kết quả phát hiện
        for box, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
            if score > 0.5:  # Ngưỡng confidence
                x1, y1, x2, y2 = box.int().cpu().numpy()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Hiển thị khung hình với các khuôn mặt được phát hiện
        cv2.imshow("Face Detection using Faster R-CNN", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Bước 2: Tạo mô hình Faster R-CNN từ mô hình pre-trained
def create_model(num_classes):
    # Tải mô hình Faster R-CNN đã huấn luyện trước
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Thay đổi lớp đầu ra để phù hợp với số lớp (face và no_face)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# Bước 3: Tạo Custom Dataset cho PyTorch từ thư mục chứa ảnh
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))

    def __getitem__(self, idx):
        # Đọc hình ảnh
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Chuyển đổi từ PIL image sang PyTorch tensor
        img = ToTensor()(img)  # Chuyển hình ảnh thành tensor
        
        # TODO: Đọc annotation từ tệp (XML, JSON,...)
        boxes = [[50, 50, 200, 200]]  # Bạn cần thay thế bằng tọa độ bounding boxes thực tế
        labels = [1]  # 1 cho face, 0 cho no_face

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return img, target

    def __len__(self):
        return len(self.imgs)

# Bước 4: Huấn luyện mô hình
def train_model(model, train_loader, val_loader, device, num_epochs=5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            # Chuyển hình ảnh sang GPU
            images = [image.to(device) for image in images]  # Bây giờ image đã là tensor
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs} completed.")

    torch.save(model.state_dict(), 'faster_rcnn_face_detection.pth')

# Bước 5: Đánh giá mô hình
def evaluate_model(model, val_loader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            # TODO: Đánh giá dựa trên outputs và targets (bạn có thể tính chính xác bounding box hoặc class).
            # Đây chỉ là ví dụ khái quát
            total += len(targets)
            # Ví dụ minh họa
            correct += 1

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Bước 6: Main program - Phát hiện khuôn mặt và Huấn luyện mô hình
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Phát hiện khuôn mặt qua webcam
    print("Start face detection from webcam using Faster R-CNN (Press 'q' to exit)...")
    model = create_model(num_classes=2)  # Số lớp: face, no_face
    detect_faces_from_webcam(model, device)

    # Huấn luyện mô hình với dataset tùy chỉnh
    print("Start training the model with the dataset...")
    train_dataset = CustomDataset(root='D:/Face_Detect_Mtcnn/Mtcnn/datasets/datasets 1/train')
    val_dataset = CustomDataset(root='D:/Face_Detect_Mtcnn/Mtcnn/datasets/datasets 1/val')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Huấn luyện mô hình
    print("Training in progress...")
    train_model(model, train_loader, val_loader, device)

    # Đánh giá mô hình
    print("Evaluating the model...")
    evaluate_model(model, val_loader, device)
