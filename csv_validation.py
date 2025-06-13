# In csv_validation.py

import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Preprocess # Use the robust Preprocess class
from retinanet import csv_eval

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple evaluation script for a RetinaNet network.')

    parser.add_argument('--csv_annotations', help='Path to CSV annotations file for validation')
    parser.add_argument('--model_path', help='Path to saved model state_dict (.pt file)', type=str)
    parser.add_argument('--class_list', help='Path to classlist csv', type=str)
    
    # MODIFIED: Add a score threshold argument for precision/recall calculation
    parser.add_argument('--iou_threshold', help='IOU threshold used for mAP', type=float, default=0.5)
    parser.add_argument('--score_threshold', help='Score threshold for Precision/Recall calculation', type=float, default=0.5)
    
    parser = parser.parse_args(args)

    dataset_val = CSVDataset(train_file=parser.csv_annotations, class_list=parser.class_list, transform=Preprocess())

    # Create model and load weights
    print("Creating model structure...")
    num_classes = dataset_val.num_classes()
    retinanet = model.efficientnet_b0_retinanet(num_classes=num_classes)
    print(f"Loading weights from {parser.model_path}...")
    retinanet.load_state_dict(torch.load(parser.model_path))

    # Setup device and eval mode
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
    retinanet.eval()
    if hasattr(retinanet, 'module'):
        retinanet.module.freeze_bn()
    else:
        retinanet.freeze_bn()

    print("Evaluating model...")
    # MODIFIED: Pass the score threshold to the evaluate function
    results = csv_eval.evaluate(
        dataset_val, 
        retinanet, 
        iou_threshold=parser.iou_threshold,
        score_threshold=parser.score_threshold
    )
    
    # MODIFIED: Print the richer results dictionary
    print("\n--- Final Metrics ---")
    for key, value in results.items():
        print(f"{key.upper()}: {value:.4f}")
    print("---------------------")


if __name__ == '__main__':
    main()