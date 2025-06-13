import torch
import numpy as np
import time
import os
import argparse

# Import necessary components from your project
from retinanet import model
from retinanet.dataloader import CSVDataset, CocoDataset, Preprocess
from pycocotools.cocoeval import COCOeval
import json

def benchmark_model(args):
    # --- 1. Load Dataset ---
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'csv':
        dataset_val = CSVDataset(
            train_file=args.csv_val,
            class_list=args.csv_classes,
            transform=Preprocess()
        )
    elif args.dataset == 'coco':
        dataset_val = CocoDataset(
            args.coco_path,
            set_name='val2017', # Assuming standard val set
            transform=Preprocess()
        )
    else:
        raise ValueError("Dataset must be 'csv' or 'coco'")

    # --- 2. Load Model ---
    print(f"Loading model from {args.model}...")
    num_classes = dataset_val.num_classes()
    
    # You might need to adjust this logic if you use different backbones
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
            
            # Time the inference
            start_time = time.time()
            torch.cuda.synchronize() # Wait for all previous GPU work to finish
            
            scores, labels, boxes = retinanet(input_tensor)
            
            torch.cuda.synchronize() # Wait for the model inference to finish
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000) # a an ms
            
            # Process outputs
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            confident_indices = np.where(scores > 0.05)[0] # Use a low threshold for mAP calculation
            
            boxes = boxes[confident_indices]
            scores = scores[confident_indices]
            labels = labels[confident_indices]

            # Change to COCO format (x, y, w, h)
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
            
            # Get the correct image_id for COCO eval
            img_id = dataset_val.image_ids[index] if args.dataset == 'coco' else index
            image_ids.append(img_id)

            for box_id in range(boxes.shape[0]):
                score = float(scores[box_id])
                label = int(labels[box_id])
                box = boxes[box_id, :]
                
                # Get COCO category_id
                if args.dataset == 'coco':
                    category_id = dataset_val.label_to_coco_label(label)
                else: # For CSV, the label is the category_id
                    category_id = label

                image_result = {
                    'image_id': img_id,
                    'category_id': category_id,
                    'score': score,
                    'bbox': box.tolist(),
                }
                results.append(image_result)
        
        print(f"Processed {index+1}/{len(dataset_val)} images", end='\r')

    print("\nInference complete. Calculating mAP...")

    if not len(results):
        print("No detections were made.")
        return

    # --- 4. Calculate COCO Metrics ---
    # To use the COCO eval tool for CSV datasets, we need a mock COCO object
    if args.dataset == 'csv':
        # Create a mock COCO ground truth object
        mock_coco_gt = {'images': [], 'annotations': [], 'categories': []}
        ann_id_counter = 1
        for i in range(len(dataset_val)):
            mock_coco_gt['images'].append({'id': i, 'height': 1, 'width': 1}) # height/width don't matter here
            annots = dataset_val.load_annotations(i)
            for ann in annots:
                bbox = [ann[0], ann[1], ann[2]-ann[0], ann[3]-ann[1]] # to x,y,w,h
                mock_coco_gt['annotations'].append({
                    'id': ann_id_counter,
                    'image_id': i,
                    'category_id': int(ann[4]),
                    'bbox': bbox,
                    'iscrowd': 0,
                    'area': bbox[2] * bbox[3]
                })
                ann_id_counter += 1
        for i in range(dataset_val.num_classes()):
            mock_coco_gt['categories'].append({'id': i, 'name': dataset_val.label_to_name(i)})
        
        from pycocotools.coco import COCO
        coco_true = COCO()
        coco_true.dataset = mock_coco_gt
        coco_true.createIndex()
    else: # For COCO dataset
        coco_true = dataset_val.coco

    coco_pred = coco_true.loadRes(results)
    
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # --- 5. Extract and Print Final Results ---
    stats = coco_eval.stats
    avg_inference_time = np.mean(inference_times)
    fps = 1000 / avg_inference_time

    print("\n--- Benchmark Results ---")
    print(f"mAP @[IoU=0.50:0.95]: {stats[0]:.4f}")
    print(f"mAP @[IoU=0.50]:      {stats[1]:.4f}")
    print(f"Average Recall @[maxDets=100]: {stats[8]:.4f}")
    print("---")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("-------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark an object detection model.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', required=True)
    parser.add_argument('--model', help='Path to model state_dict (.pt) file', required=True)
    
    # Args for CSV dataset
    parser.add_argument('--csv_val', help='Path to file containing validation annotations')
    parser.add_argument('--csv_classes', help='Path to file containing class list')

    # Args for COCO dataset
    parser.add_argument('--coco_path', help='Path to COCO directory')

    args = parser.parse_args()
    benchmark_model(args)