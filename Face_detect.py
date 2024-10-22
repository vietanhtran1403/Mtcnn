import cv2
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Bước 1: Phát hiện khuôn mặt từ webcam bằng MTCNN
def detect_faces_from_webcam():
    detector = MTCNN()  # Khởi tạo MTCNN để phát hiện khuôn mặt
    cap = cv2.VideoCapture(0)  # Mở webcam

    if not cap.isOpened():
        print("Unable to access webcam!")
        return

    while True:
        ret, frame = cap.read()  # Đọc khung hình từ webcam
        if not ret:
            print("Unable to read frames from webcam!")
            break

        # Phát hiện khuôn mặt trong khung hình
        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, width, height = face['box']
            # Vẽ hình chữ nhật quanh khuôn mặt được phát hiện
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Hiển thị khung hình với các khuôn mặt được phát hiện
        cv2.imshow("Face Detection", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Bước 2: Tạo mô hình MobileNetV2 cho phát hiện khuôn mặt
def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)  # num_classes = 2 (face, no_face)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Bước 3: Chuẩn bị dữ liệu huấn luyện và kiểm tra từ dataset
def prepare_data(train_path, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    return train_generator, validation_generator

# Bước 4: Huấn luyện mô hình với Early Stopping
def train_model(model, train_generator, validation_generator, epochs=5):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        steps_per_epoch=260,
        validation_steps=140,
        callbacks=[early_stopping])
    
    model.save('face_detection_model.h5')

# Bước 5: Đánh giá mô hình với tập test
def evaluate_model(test_path):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical')

    model = tf.keras.models.load_model('face_detection_model.h5')
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Bước 6: Main program - Phát hiện khuôn mặt và Huấn luyện mô hình
if __name__ == "__main__":
    # Phát hiện khuôn mặt qua webcam
    print("Start face detection from webcam (Press 'q' to exit)...")
    detect_faces_from_webcam()

    # Huấn luyện mô hình với dataset
    print("Start training the model with the dataset...")
    train_path = 'D:/Face_Detect_Mtcnn/Mtcnn/datasets/datasets 1/train'  # Thay bằng đường dẫn thực tế tới dữ liệu huấn luyện của bạn
    test_path = 'D:/Face_Detect_Mtcnn/Mtcnn/datasets/datasets 1/val'  # Thay bằng đường dẫn thực tế tới dữ liệu kiểm tra

    num_classes = 2  # Số lớp: face, no_face
    model = create_model(num_classes)

    # Chuẩn bị dữ liệu huấn luyện và validation
    train_generator, validation_generator = prepare_data(train_path)

    # Huấn luyện mô hình
    print("Model training in progress...")
    train_model(model, train_generator, validation_generator)

    # Đánh giá mô hình
    print("Evaluating the model...")
    evaluate_model(test_path)
