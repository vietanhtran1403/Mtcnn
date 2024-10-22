import os

# Đếm số lượng ảnh trong thư mục train
train_path = 'D:/Face_Detect_Mtcnn/Mtcnn/datasets/datasets 1/train'
val_path = 'D:/Face_Detect_Mtcnn/Mtcnn/datasets/datasets 1/val'

def count_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                count += 1
    return count

# Đếm số lượng ảnh
num_train_images = count_images(train_path)
num_val_images = count_images(val_path)

print(f'Number of images in the training set: {num_train_images}')
print(f'Number of images in the Val set: {num_val_images}')
