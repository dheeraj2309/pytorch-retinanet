import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
import skimage.transform
from IPython.display import display, Image
from torchvision.ops import nms

# We need the model definition and UnNormalizer
from retinanet import model
from retinanet.dataloader import UnNormalizer

# --- Helper function to load classes ---
def load_classes(csv_path):
    classes = {}
    labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line, row in enumerate(reader):
            try:
                class_name, class_id = row
            except ValueError:
                raise ValueError(f'line {line+1}: format should be \'class_name,class_id\'')
            
            class_id = int(class_id)
            classes[class_name] = class_id
            labels[class_id] = class_name
    return classes, labels

# --- Main function to process and visualize a single image ---
def visualize_single_image(image_path, model_path, class_list_path, score_threshold=0.4,nms_threshold=0.5):
    """
    Loads a model, processes a single image, runs inference, and RETURNS the annotated image.
    """
    # --- 1. Load Classes and Model ---
    # print("Loading classes and model...") # Comment out for cleaner subplot output
    classes, labels = load_classes(class_list_path)
    num_classes = len(classes)
    
    # Create a model instance
    retinanet = model.efficientnet_b0_retinanet(num_classes=num_classes)
    
    # Load the saved weights (state_dict)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    retinanet.load_state_dict(state_dict)
    
    # Set up device and eval mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    retinanet = retinanet.to(device)
    retinanet.eval()
    # print(f"Model loaded on {device}.")

    # --- 2. Load and Pre-process the Image ---
    image_orig = cv2.imread(image_path)
    if image_orig is None:
        print(f"Error: Could not read image at {image_path}")
        return None # Return None on error

    image_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    
    # --- Manual Pre-processing ---
    # ... (this logic is unchanged)
    rows, cols, cns = image_rgb.shape
    min_side, max_side = 608, 1024
    smallest_side = min(rows, cols)
    scale = min_side / smallest_side
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    image_resized = skimage.transform.resize(image_rgb, (int(round(rows * scale)), int(round(cols * scale))), preserve_range=True, mode='constant').astype(np.float32)
    rows, cols, cns = image_resized.shape
    pad_w = 32 - rows % 32 if rows % 32 != 0 else 0
    pad_h = 32 - cols % 32 if cols % 32 != 0 else 0
    new_image_padded = np.zeros((rows + pad_w, cols + pad_h, cns), dtype=np.float32)
    new_image_padded[:rows, :cols, :] = image_resized
    image_tensor = torch.from_numpy(new_image_padded).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized_image = (image_tensor / 255.0 - mean) / std

    # --- 3. Run Inference ---
    with torch.no_grad():
        input_tensor = normalized_image.to(device).float().unsqueeze(dim=0)
        scores, pred_labels, pred_boxes = retinanet(input_tensor)
        scores, pred_labels, pred_boxes = scores.cpu(), pred_labels.cpu(), pred_boxes.cpu()
        
    # --- 4. Draw Detections ---
    confident_indices = torch.where(scores > score_threshold)[0]
    scores = scores[confident_indices]
    pred_labels = pred_labels[confident_indices]
    pred_boxes = pred_boxes[confident_indices]

    # This is the NMS loop
    final_boxes = []
    final_labels = []
    final_scores = []

    # NMS is applied on a per-class basis
    for class_id in range(num_classes):
        # Get all detections for the current class
        print(type(pred_labels))
        class_indices = torch.where(pred_labels == class_id)[0]
        
        if len(class_indices) == 0:
            continue
            
        class_boxes = pred_boxes[class_indices]
        class_scores = scores[class_indices]
        
        # Apply Non-Max Suppression
        # nms() returns the indices of the boxes to keep
        keep_indices = nms(class_boxes, class_scores, iou_threshold=nms_threshold)
        
        # Store the kept boxes, labels, and scores
        final_boxes.append(class_boxes[keep_indices])
        final_labels.append(torch.full_like(class_scores[keep_indices], fill_value=class_id, dtype=torch.int))
        final_scores.append(class_scores[keep_indices])

    # Concatenate the results from all classes
    if len(final_boxes) > 0:
        pred_boxes_final = torch.cat(final_boxes, dim=0)
        pred_labels_final = torch.cat(final_labels, dim=0)
        scores_final = torch.cat(final_scores, dim=0)
        
        # MODIFIED: Convert to NumPy arrays AFTER NMS is fully complete
        pred_boxes_final = pred_boxes_final.numpy()
        pred_labels_final = pred_labels_final.numpy()
        scores_final = scores_final.numpy()
    else: # If no boxes were kept after NMS
        pred_boxes_final, pred_labels_final, scores_final = np.array([]), np.array([]), np.array([])
    
    import matplotlib.pyplot as plt
    cmap = plt.colormaps.get_cmap('hsv')
    colors = (cmap(np.linspace(0, 1, num_classes))[:, :3] * 255).astype(np.uint8)

    detection_count = len(pred_boxes)
    print(f'Total detections after NMS: {detection_count}')
    
    # The loop now iterates over the CLEANED detections
    for i in range(detection_count):
        box, label_id, score = pred_boxes_final[i, :], pred_labels_final[i], scores_final[i]
        
        class_name = labels[label_id]
        color = colors[label_id].tolist()
        
        caption = f'{class_name} {score:.2f}'
        
        x1, y1, x2, y2 = map(int, box / scale)
        
        cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=color, thickness=2)
        (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image_orig, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        cv2.putText(image_orig, caption, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    return cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training on a single image.')
    parser.add_argument('--image_path', help='Path to a single image file')
    parser.add_argument('--model_path', help='Path to model state_dict')
    parser.add_argument('--class_list', help='Path to CSV file listing class names')
    parser.add_argument('--threshold', help='Score threshold for detections', type=float, default=0.4)
    args = parser.parse_args()

    visualize_single_image(args.image_path, args.model_path, args.class_list, args.threshold)