import torch
import numpy as np
import time
import os
import argparse
import skimage.transform # Import for resizing

# Import necessary components from your project
from retinanet import model
from retinanet.dataloader import CSVDataset, CocoDataset # We no longer import Preprocess from here
from pycocotools.coco import COCOeval
import json

# =============================================================================
# NEW: Define the Preprocess class directly inside this script.
# This makes the script self-contained and removes the ImportError.
# This class mimics the Resizer -> Normalizer chain from your training validation set.
# =============================================================================
class Preprocess(object):
    """
    A self-contained class for validation/inference pre-processing.
    It handles resizing, normalization, and tensor conversion consistently.
    """
    def __init__(self, min_side=608, max_side=1024):
        self.min_side = min_side
        self.max_side = max_side
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, sample):
        # 1. Unpack initial sample {'img': np.array, 'annot': np.array}
        image, annots = sample['img'], sample['annot']

        # 2. Perform Resizing
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        scale_factor = self.min_side / smallest_side
        largest_side = max(rows, cols)

        if largest_side * scale_factor > self.max_side:
            scale_factor = self.max_side / largest_side
        
        image_resized = skimage.transform.resize(
            image, 
            (int(round(rows * scale_factor)), int(round(cols * scale_factor))),
            preserve_range=True, mode='constant'
        ).astype(np.float32)

        rows, cols, cns = image_resized.shape
        pad_w = 32 - rows % 32 if rows % 32 != 0 else 0
        pad_h = 32 - cols % 32 if cols % 32 != 0 else 0

        new_image_padded = np.zeros((rows + pad_w, cols + pad_h, cns), dtype=np.float32)
        new_image_padded[:rows, :cols, :] = image_resized
        
        annots[:, :4] *= scale_factor
        
        # 3. Perform Normalization and Tensor Conversion
        image_tensor = torch.from_numpy(new_image_padded)
        annot_tensor = torch.from_numpy(annots)

        image_tensor = image_tensor.permute(2, 0, 1)
        normalized_image = (image_tensor / 255.0 - self.mean) / self.std

        # 4. Return the final dictionary
        return {
            'img': normalized_image,
            'annot': annot_tensor,
            'scale': scale_factor
        }
# =============================================================================
# End of Preprocess class definition
# =============================================================================

def benchmark_model(args):
    # --- 1. Load Dataset ---
    print(f"Loading {args.dataset} dataset...")
    # MODIFIED: The Preprocess class is now defined locally in this script
    if args.dataset == 'csv':
        dataset_val = CSVDataset(
            train_file=args.csv_val,
            class_list=args.csv_classes,
            transform=Preprocess()
        )
    elif args.dataset == 'coco':
        # Assuming you have a COCO dataset to test with
        dataset_val = CocoDataset(
            args.coco_path,
            set_name='val2017',
            transform=Preprocess()
        )
    else:
        raise ValueError("Dataset must be 'csv' or 'coco'")

    # --- 2. Load Model ---
    print(f"Loading model from {args.model}...")
    num_classes = dataset_val.num_classes()
    
    # This logic assumes efficientnet-b0. Adjust if you benchmark other models.
    retinanet = model.efficientnet_b0_retinanet(num_classes=num_classes)
    
    state_dict = torch.load(args.model, map_location=torch.device('cpu'))
    retinanet.load_state_dict(state_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    retinanet = retinanet.to(device)
    retinanet.eval()
    print(f"Model loaded on {device}.")

    # --- 3. Run Evaluation and Collect Metrics ---
    results = []
    image_ids = []
    inference_times = []

    print("Running inference on validation set...")
    for index in range(len(dataset_val)):
        data = dataset_val[index]
        
        with torch.no_grad():
            input_tensor = data['img'].to(device).float().unsqueeze(0)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            scores, labels, boxes = retinanet(input_tensor)
            
            torch.cuda.synchronize()
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)
            
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            confident_indices = np.where(scores > 0.05)[0]
            
            boxes = boxes[confident_indices]
            scores = scores[confident_indices]
            labels = labels[confident_indices]

            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
            
            img_id = dataset_val.image_ids[index] if hasattr(dataset_val, 'image_ids') else index
            image_ids.append(img_id)

            for box_id in range(boxes.shape[0]):
                score, label, box = float(scores[box_id]), int(labels[box_id]), boxes[box_id, :]
                
                category_id = dataset_val.label_to_coco_label(label) if hasattr(dataset_val, 'label_to_coco_label') else label
                
                image_result = {'image_id': img_id, 'category_id': category_id, 'score': score, 'bbox': box.tolist()}
                results.append(image_result)
        
        print(f"Processed {index+1}/{len(dataset_val)} images", end='\r')

    print("\nInference complete. Calculating mAP...")

    if not len(results):
        print("No detections were made.")
        return

    # --- 4. Calculate COCO Metrics ---
    if args.dataset == 'csv':
        mock_coco_gt = {'images': [], 'annotations': [], 'categories': []}
        ann_id_counter = 1
        for i in range(len(dataset_val)):
            mock_coco_gt['images'].append({'id': i, 'height': 1, 'width': 1})
            annots = dataset_val.load_annotations(i)
            for ann in annots:
                bbox = [ann[0], ann[1], ann[2]-ann[0], ann[3]-ann[1]]
                mock_coco_gt['annotations'].append({'id': ann_id_counter, 'image_id': i, 'category_id': int(ann[4]), 'bbox': bbox, 'iscrowd': 0, 'area': bbox[2] * bbox[3]})
                ann_id_counter += 1
        for i in range(dataset_val.num_classes()):
            mock_coco_gt['categories'].append({'id': i, 'name': dataset_val.label_to_name(i)})
        
        coco_true = COCO()
        coco_true.dataset = mock_coco_gt
        coco_true.createIndex()
    else:
        coco_true = dataset_val.coco

    coco_pred = coco_true.loadRes(results)
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # --- 5. Extract and Print Final Results ---
    stats = coco_eval.stats
    # To get a more stable inference time, ignore the first few readings (GPU warm-up)
    if len(inference_times) > 5:
        avg_inference_time = np.mean(inference_times[5:])
    else:
        avg_inference_time = np.mean(inference_times)
        
    fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0

    print("\n--- Benchmark Results ---")
    print(f"mAP @[IoU=0.50:0.95]:         {stats[0]:.4f}")
    print(f"mAP @[IoU=0.50]:              {stats[1]:.4f}")
    print(f"Average Recall @[maxDets=100]: {stats[8]:.4f}")
    print("---")
    print(f"Average Inference Time:      {avg_inference_time:.2f} ms")
    print(f"Frames Per Second (FPS):       {fps:.2f}")
    print("-------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark an object detection model.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', required=True)
    parser.add_argument('--model', help='Path to model state_dict (.pt) file', required=True)
    parser.add_argument('--csv_val', help='Path to file containing validation annotations')
    parser.add_argument('--csv_classes', help='Path to file containing class list')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    args = parser.parse_args()
    benchmark_model(args)