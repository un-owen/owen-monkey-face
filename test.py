import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_test_set(model_path, test_dir):
    model = YOLO(model_path)

    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    true_labels = []
    pred_labels = []
    image_paths = []
    
    for cls_name in tqdm(classes, desc="Processing Classes"):
        cls_dir = os.path.join(test_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(cls_dir, img_name)
            results = model(img_path)
            pred_cls = results[0].probs.top1
            pred_cls_name = results[0].names[pred_cls]
            
            true_labels.append(class_to_idx[cls_name])
            pred_labels.append(class_to_idx[pred_cls_name])
            image_paths.append(img_path)

    accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=list(class_to_idx.values()))
    cls_report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=classes,
        digits=4
    )
    
    print("\n===== Evaluation Results =====")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cls_report)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": cls_report,
        "class_names": classes,
        "details": list(zip(image_paths, true_labels, pred_labels))
    }

if __name__ == "__main__":

    model_path = "runs/classify/monkey/weights/best.pt"
    test_dir = "data/test"
    
    results = evaluate_test_set(model_path, test_dir)