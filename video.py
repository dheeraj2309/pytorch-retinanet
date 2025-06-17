# inference_video_standalone.py

import cv2
import torch
import numpy as np
import argparse
import time
import os
import sys

# Add the project's root directory to the Python path to allow importing the 'retinanet' module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from retinanet import model as retinanet_model

# --- Helper Functions and Classes ---

def _load_class_labels(csv_path):
    """
    Parses a class list CSV file.
    Expected format: class_name,class_id (e.g., car,0)
    Returns a dictionary mapping {id: 'name'}.
    """
    labels = {}
    try:
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    name, class_id = line.split(',')
                    labels[int(class_id)] = name
    except FileNotFoundError:
        print(f"Error: Class CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error parsing class CSV file: {e}")
        return None
        
    print(f"Loaded {len(labels)} classes from {os.path.basename(csv_path)}.")
    return labels

class VideoPreprocessor:
    """
    Handles resizing, padding, and normalization for a single video frame.
    """
    def __init__(self, min_side=608, max_side=1024):
        self.min_side = min_side
        self.max_side = max_side
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, image_np):
        rows, cols, _ = image_np.shape
        smallest_side = min(rows, cols)
        scale_factor = self.min_side / smallest_side
        largest_side = max(rows, cols)

        if largest_side * scale_factor > self.max_side:
            scale_factor = self.max_side / largest_side

        image_resized = cv2.resize(image_np, (int(cols * scale_factor), int(rows * scale_factor)))
        rows_r, cols_r, _ = image_resized.shape

        pad_h = 32 - rows_r % 32 if rows_r % 32 != 0 else 0
        pad_w = 32 - cols_r % 32 if cols_r % 32 != 0 else 0

        new_image_padded = np.zeros((rows_r + pad_h, cols_r + pad_w, 3), dtype=np.float32)
        new_image_padded[:rows_r, :cols_r, :] = image_resized

        image_tensor = torch.from_numpy(new_image_padded)
        image_tensor = image_tensor.permute(2, 0, 1)
        normalized_image = (image_tensor / 255.0 - self.mean) / self.std

        return normalized_image, scale_factor

# --- Main Inference Function ---

def run_video_detection(args):
    # --- 1. Setup Model and Device ---
    print("Loading model and class definitions...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    class_labels = _load_class_labels(args.csv_classes)
    if class_labels is None:
        return
    num_classes = len(class_labels)

    detector = retinanet_model.efficientnet_b0_retinanet(num_classes=num_classes)
    
    try:
        detector.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {args.model_path}")
        return
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        return

    detector = detector.to(device)
    detector.eval()

    # --- 2. Setup Video I/O ---
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    print(f"Processing video. Output will be saved to {args.output_path}")

    # --- 3. Process Video Frames ---
    preprocessor = VideoPreprocessor()
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    colors = (np.random.rand(num_classes, 3) * 255).astype(np.uint8)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- MODIFICATION START: Initialize vehicle count for the frame ---
            vehicle_count = 0
            # --- MODIFICATION END ---
            
            frame_count += 1
            if frame_count % args.nth_frame != 0:
                out.write(frame)
                continue

            start_time = time.time()
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, scale = preprocessor(frame_rgb.astype(np.float32))
            
            scores, labels, boxes = detector(processed_frame.to(device).float().unsqueeze(dim=0))
            
            scores, labels, boxes = scores.cpu(), labels.cpu(), boxes.cpu() / scale 
            
            confident_indices = torch.where(scores > args.score_threshold)[0]
            
            # --- MODIFICATION START: Update vehicle count based on detections ---
            # We assume your model is trained on a single "vehicle" class or similar.
            # If you have multiple classes (car, truck, bus), they will all be counted.
            vehicle_count = len(confident_indices)
            # --- MODIFICATION END ---
            
            for i in confident_indices:
                box = boxes[i]
                label_id = labels[i].item()
                score = scores[i].item()
                
                x1, y1, x2, y2 = map(int, box)
                
                class_name = class_labels.get(label_id, f"ID:{label_id}")
                color = colors[label_id % num_classes].tolist()
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                caption = f'{class_name}: {score:.2f}'
                (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                cv2.putText(frame, caption, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            end_time = time.time()
            processing_fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else float('inf')
            
            # --- MODIFICATION START: Draw the counter on the frame ---
            counter_text = f"Vehicles Detected: {vehicle_count}"
            text_size, _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_TRIPLEX, 1, 2)
            text_w, text_h = text_size
            
            # Position in the top-right corner
            text_x = width - text_w - 20 # 20 pixels padding from the right edge
            text_y = text_h + 20        # 20 pixels padding from the top edge
            
            # Draw a semi-transparent background rectangle for better readability
            sub_img = frame[text_y - text_h - 10:text_y + 10, text_x - 10:text_x + text_w + 10]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
            frame[text_y - text_h - 10:text_y + 10, text_x - 10:text_x + text_w + 10] = res
            
            # Draw the text
            cv2.putText(frame, counter_text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
            # --- MODIFICATION END ---

            print(f"Processing Frame {frame_count}/{total_frames} | Detections: {len(confident_indices)} | FPS: {processing_fps:.2f}")
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RetinaNet detection on a video using separate model and class files.')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained RetinaNet model weights file (.pt).')
    parser.add_argument('--csv_classes', type=str, required=True, help='Path to the CSV file mapping class names to IDs (e.g., car,0).')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    
    parser.add_argument('--output_path', type=str, default='output_detection.mp4', help='Path to save the output video.')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Confidence score threshold for showing detections.')
    parser.add_argument('--nth_frame', type=int, default=1, help='Process every N-th frame for faster inference. Default is 1 (every frame).')

    args = parser.parse_args()
    run_video_detection(args)