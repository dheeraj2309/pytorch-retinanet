# inference_video.py

import cv2
import torch
import numpy as np
import argparse
import time
import os
import sys

# --- MODIFICATION FOR VENDORED DEPENDENCY ---

# 1. Get the absolute path to the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construct the path to the directory that CONTAINS the 'torchreid' package.
#    Based on your path, this is the 'Deep-ReID' folder.
#    We create a relative path from our script's location.
path_to_reid_repo = os.path.join(script_dir, 'external', 'Deep-ReID')

# 3. Add this path to the beginning of Python's system path.
#    Now, when Python looks for a package named 'torchreid', it will
#    search inside our '.../external/Deep-ReID/' folder and find it.
sys.path.insert(0, path_to_reid_repo)

# --- END OF MODIFICATION ---


# With the path correctly set, these imports will now work.
from retinanet import model as retinanet_model
from feature_extractor import FeatureExtractor


# --- Helper Classes ---

class VideoPreprocessor:
    """
    A self-contained class for inference pre-processing.
    It handles resizing, normalization, and tensor conversion for a single image frame.
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

        pad_w = 32 - rows_r % 32 if rows_r % 32 != 0 else 0
        pad_h = 32 - cols_r % 32 if cols_r % 32 != 0 else 0

        new_image_padded = np.zeros((rows_r + pad_w, cols_r + pad_h, 3), dtype=np.float32)
        new_image_padded[:rows_r, :cols_r, :] = image_resized

        image_tensor = torch.from_numpy(new_image_padded)
        image_tensor = image_tensor.permute(2, 0, 1)
        normalized_image = (image_tensor / 255.0 - self.mean) / self.std

        return normalized_image, scale_factor

def _compute_cosine_similarity(features1, features2):
    """Computes cosine similarity between two sets of feature vectors."""
    features1 = torch.nn.functional.normalize(features1, p=2, dim=1)
    features2 = torch.nn.functional.normalize(features2, p=2, dim=1)
    similarity_matrix = torch.mm(features1, features2.T)
    return similarity_matrix

# --- Main Inference Function ---

def run_inference(args):
    # --- 1. Setup Models and Data ---
    print("Loading models and setting up...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the checkpoint which contains model weights and class labels
    checkpoint = torch.load(args.detector_model, map_location=device)

    # Extract class labels from the checkpoint
    class_labels = checkpoint['class_labels']
    num_classes = len(class_labels)
    print(f"Found {num_classes} classes in the model file.")

    # Load RetinaNet detector
    detector = retinanet_model.efficientnet_b0_retinanet(num_classes=num_classes)
    detector.load_state_dict(checkpoint['model_state_dict'])
    detector = detector.to(device)
    detector.eval()

    # Load Re-ID Feature Extractor
    reid_extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=args.reid_model,
        device=str(device)
    )

    # Setup preprocessor
    preprocessor = VideoPreprocessor()

    # --- 2. Tracking Initialization ---
    tracked_objects = {}
    next_track_id = 0
    colors = (np.random.rand(num_classes, 3) * 255).astype(np.uint8)

    # --- 3. Video Processing ---
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    frame_count = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % args.nth_frame != 0:
                out.write(frame) # Write the unprocessed frame to keep the video smooth
                continue

            start_time = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, scale = preprocessor(frame_rgb.astype(np.float32))
            
            scores, labels, boxes = detector(processed_frame.to(device).float().unsqueeze(dim=0))
            
            scores, labels, boxes = scores.cpu(), labels.cpu(), boxes.cpu() / scale
            
            confident_indices = torch.where(scores > args.score_threshold)[0]
            if len(confident_indices) == 0:
                out.write(frame)
                continue
            
            scores, labels, boxes = scores[confident_indices], labels[confident_indices], boxes[confident_indices]

            detection_crops = []
            valid_boxes_indices = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                if x1 >= x2 or y1 >= y2: continue
                crop = frame_rgb[y1:y2, x1:x2]
                detection_crops.append(crop)
                valid_boxes_indices.append(i)
            
            if not detection_crops:
                out.write(frame)
                continue

            # Filter the original boxes and labels to match the valid crops
            boxes = boxes[valid_boxes_indices]
            labels = labels[valid_boxes_indices]
            
            current_features = reid_extractor(detection_crops)
            
            matched_indices = set()
            newly_tracked_objects = {}

            if len(tracked_objects) > 0:
                track_ids = list(tracked_objects.keys())
                track_features = torch.stack([t['feature'] for t in tracked_objects.values()]).squeeze(1).to(device)
                similarity_matrix = _compute_cosine_similarity(current_features, track_features)
                
                for i in range(len(detection_crops)):
                    best_match_score, best_match_idx = torch.max(similarity_matrix[i], dim=0)
                    if best_match_score > args.reid_threshold:
                        track_id = track_ids[best_match_idx]
                        if track_id not in newly_tracked_objects:
                            newly_tracked_objects[track_id] = {
                                'feature': current_features[i],
                                'bbox': boxes[i],
                                'class_name': class_labels[labels[i].item()],
                                'age': 0
                            }
                            matched_indices.add(i)
                            similarity_matrix[:, best_match_idx] = -1

            for i in range(len(detection_crops)):
                if i not in matched_indices:
                    newly_tracked_objects[next_track_id] = {
                        'feature': current_features[i],
                        'bbox': boxes[i],
                        'class_name': class_labels[labels[i].item()],
                        'age': 0
                    }
                    next_track_id += 1

            for track_id in list(tracked_objects.keys()):
                if track_id not in newly_tracked_objects:
                    tracked_objects[track_id]['age'] += 1
                    if tracked_objects[track_id]['age'] > args.max_age:
                        del tracked_objects[track_id]
                    else:
                        newly_tracked_objects[track_id] = tracked_objects[track_id]
            
            tracked_objects = newly_tracked_objects

            for track_id, track_data in tracked_objects.items():
                if track_data['age'] > 0 and args.hide_lost_tracks: continue

                box, class_name = track_data['bbox'], track_data['class_name']
                x1, y1, x2, y2 = map(int, box)
                
                class_id = [k for k, v in class_labels.items() if v == class_name][0]
                color = colors[class_id].tolist()
                caption = f'ID: {track_id} {class_name}'
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                cv2.putText(frame, caption, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            end_time = time.time()
            processing_fps = 1 / (end_time - start_time)
            print(f"Frame {frame_count}: {len(tracked_objects)} active tracks. FPS: {processing_fps:.2f}")

            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output video saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RetinaNet detection and Re-ID tracking on a video.')
    
    # Paths
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output_path', type=str, default='output_tracked.mp4', help='Path to save the output video.')
    parser.add_argument('--detector_model', type=str, required=True, help='Path to the self-contained RetinaNet model file (.pt).')
    parser.add_argument('--reid_model', type=str, required=True, help='Path to the pre-trained Re-ID model (e.g., osnet_x1_0.pth).')
    
    # Parameters
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Confidence score threshold for detections.')
    parser.add_argument('--reid_threshold', type=float, default=0.6, help='Cosine similarity threshold for Re-ID matching.')
    parser.add_argument('--nth_frame', type=int, default=1, help='Process every N-th frame for faster inference.')
    parser.add_argument('--max_age', type=int, default=30, help='Maximum number of frames a track can be lost before being deleted.')
    parser.add_argument('--hide_lost_tracks', action='store_true', help='If set, do not draw boxes for tracks that are not visible in the current frame.')

    args = parser.parse_args()
    run_inference(args)