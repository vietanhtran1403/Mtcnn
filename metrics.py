import time
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Hàm để tính toán Precision, Recall và Speed (Inference Time)
def evaluate_metrics(model, test_generator, device):
    all_true_labels = []
    all_pred_labels = []
    
    total_inference_time = 0
    total_images = 0

    for images, labels in test_generator:
        start_time = time.time()
        
        # Dự đoán kết quả
        preds = model.predict(images)
        
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        total_images += len(images)
        
        # Lấy nhãn dự đoán
        pred_labels = np.argmax(preds, axis=1)
        all_pred_labels.extend(pred_labels)
        all_true_labels.extend(labels)

    precision = precision_score(all_true_labels, all_pred_labels, average='weighted')
    recall = recall_score(all_true_labels, all_pred_labels, average='weighted')
    avg_inference_time = total_inference_time / total_images

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Average Inference Time per Image: {avg_inference_time:.4f} seconds")
