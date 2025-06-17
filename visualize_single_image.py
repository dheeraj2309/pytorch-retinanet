import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
import skimage.transform
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt

# It's better practice to have all imports at the top
from retinanet import model
from retinanet.dataloader import UnNormalizer

# --- Helper function to load classes ---
def load_classes(csv_path):
    """Parses a class list CSV and returns a mapping from ID to name."""
    labels = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for line, row in enumerate(reader):
                class_name, class_id = row
                labels[int(class_id)] = class_name
    except FileNotFoundError:
        print(f"Error: Class list file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error parsing class list file '{csv_path}': {e}")
        return None
    return labels

# --- Main function to process and visualize a single image ---
def visualize_single_image(image_path, model_path, class_list_path, score_threshold=0.4):
    """
    Loads a model, processes a single image, runs inference, and returns the annotated image.
    """
    # --- 1. Load Classes and Model ---
    labels = load_classes(class_list_path)
    if labels is None:
        return None
    num_classes = len(labels)
    
    print("Loading model...")
    retinanet = model.efficientnet_b0_retinanet(num_classes=num_classes)
    
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        retinanet.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    retinanet = retinanet.to(device)
    retinanet.eval()
    print(f"Model loaded and running on {device}.")

    # --- 2. Load and Pre-process the Image ---
    image_orig = cv2.imread(image_path)
    if image_orig is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    
    # Pre-processing logic to match training
    rows, cols, _ = image_rgb.shape
    min_side, max_side = 608, 1024
    smallest_side = min(rows, cols)
    scale = min_side / smallest_side
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    image_resized = skimage.transform.resize(image_rgb, (int(round(rows * scale)), int(round(cols * scale))), preserve_range=True, mode='constant').astype(np.float32)
    rows_r, cols_r, _ = image_resized.shape
    pad_w = 32 - rows_r % 32 if rows_r % 32 != 0 else 0
    pad_h = 32 - cols_r % 32 if cols_r % 32 != 0 else 0
    new_image_padded = np.zeros((rows_r + pad_w, cols_r + pad_h, 3), dtype=np.float32)
    new_image_padded[:rows_r, :cols_r, :] = image_resized
    image_tensor = torch.from_numpy(new_image_padded).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized_image = (image_tensor / 255.0 - mean) / std

    # --- 3. Run Inference ---
    with torch.no_grad():
        input_tensor = normalized_image.to(device).float().unsqueeze(dim=0)
        
        # The model in eval mode already performs NMS and gives final results.
        # Output is [final_scores, final_labels, final_boxes]
        scores, pred_labels, pred_boxes = retinanet(input_tensor)
        
        # Move results to CPU and convert to NumPy arrays
        scores = scores.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        pred_boxes = pred_boxes.cpu().numpy()
        
    # --- 4. Filter Detections by Score (NMS is already done) ---
    confident_indices = np.where(scores > score_threshold)[0]
    
    final_boxes = pred_boxes[confident_indices]
    final_labels = pred_labels[confident_indices]
    final_scores = scores[confident_indices]
    
    # --- 5. Visualize Final Detections ---
    cmap = plt.colormaps.get_cmap('hsv')
    colors = (cmap(np.linspace(0, 1, num_classes))[:, :3] * 255).astype(np.uint8)

    detection_count = len(final_boxes)
    print(f'Found {detection_count} detections with score > {score_threshold}')
    
    # Create a copy to draw on, preserving the original
    image_annotated = image_orig.copy()

    for i in range(detection_count):
        box, label_id, score = final_boxes[i, :], final_labels[i], final_scores[i]
        
        # The boxes are for the resized image, so we scale them back to the original image size.
        box /= scale
        x1, y1, x2, y2 = map(int, box)
        
        class_name = labels.get(label_id, f"ID:{label_id}")
        color = colors[label_id % num_classes].tolist()
        
        caption = f'{class_name} {score:.2f}'
        
        cv2.rectangle(image_annotated, (x1, y1), (x2, y2), color=color, thickness=2)
        (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image_annotated, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        cv2.putText(image_annotated, caption, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    return image_annotated

# --- The __main__ block for command-line execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for visualizing RetinaNet detections on a single image.')
    parser.add_argument('--image_path', help='Path to a single image file', required=True)
    parser.add_argument('--model_path', help='Path to model state_dict file (.pt)', required=True)
    parser.add_argument('--class_list', help='Path to CSV file listing class names (e.g., car,0)', required=True)
    parser.add_argument('--output_path', help='Path to save the annotated image. If not provided, image will be displayed only.', default=None)
    parser.add_argument('--threshold', help='Score threshold for detections', type=float, default=0.4)
    args = parser.parse_args()

    # Run the main visualization function
    annotated_image = visualize_single_image(args.image_path, args.model_path, args.class_list, args.threshold)

    if annotated_image is not None:
        if args.output_path:
            # OpenCV expects BGR format for saving, but our function returns BGR already
            # (since it draws on the original cv2.imread() result)
            cv2.imwrite(args.output_path, annotated_image)
            print(f"Annotated image saved to: {args.output_path}")
        
        # For displaying in notebooks like Jupyter or Kaggle
        try:
            # Convert BGR to RGB for correct color display in notebooks
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            _, img_encoded = cv2.imencode('.png', annotated_image_rgb)
            display(IPImage(data=img_encoded.tobytes()))
        except NameError:
            # Fallback for standard Python environments without IPython
            print("To display the image, run this in a Jupyter/IPython environment or provide an --output_path.")
            # Or uncomment below to display in a separate window
            # cv2.imshow('Detections', annotated_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()