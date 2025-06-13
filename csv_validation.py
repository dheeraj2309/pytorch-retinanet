import argparse
import torch
from torchvision import transforms

from retinanet import model
# MODIFIED: Import the correct dataloader classes
from retinanet.dataloader import CSVDataset, Preprocess
from retinanet import csv_eval

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple evaluation script for a RetinaNet network.')

    # MODIFIED: Renamed for clarity, though not required
    parser.add_argument('--csv_annotations', help='Path to CSV annotations file for validation')
    parser.add_argument('--model_path', help='Path to saved model state_dict (.pt file)', type=str)
    # parser.add_argument('--images_path',help='Path to images directory',type=str) # This is not used as path is in CSV
    parser.add_argument('--class_list', help='Path to classlist csv', type=str)
    parser.add_argument('--iou_threshold', help='IOU threshold used for evaluation', type=str, default='0.5')
    
    parser = parser.parse_args(args)

    # MODIFIED: Use the robust Preprocess class for consistency
    dataset_val = CSVDataset(train_file=parser.csv_annotations, class_list=parser.class_list, transform=Preprocess())

    # --- THE MAIN FIX IS HERE ---
    
    # 1. Create an instance of the model architecture first.
    #    The num_classes must match the model you trained.
    print("Creating model structure...")
    num_classes = dataset_val.num_classes()
    # This assumes efficientnet-b0. Change if you used a different backbone.
    retinanet = model.efficientnet_b0_retinanet(num_classes=num_classes)

    # 2. Load the state_dict (the weights) into the model instance.
    print(f"Loading weights from {parser.model_path}...")
    retinanet.load_state_dict(torch.load(parser.model_path))
    # ----------------------------

    # Now, `retinanet` is a proper torch.nn.Module object
    use_gpu = True
    if use_gpu and torch.cuda.is_available():
        retinanet = retinanet.cuda()

    # The DataParallel wrapper is generally used for multi-GPU training.
    # For evaluation, it's often simpler to run on a single GPU without it.
    # If you must use it, apply it after moving the model to the GPU.
    # if torch.cuda.is_available():
    #     retinanet = torch.nn.DataParallel(retinanet).cuda()

    # Set the model to evaluation mode
    retinanet.eval()
    
    # The freeze_bn() call is for training, not strictly necessary for eval, but harmless.
    if hasattr(retinanet, 'module'):
        retinanet.module.freeze_bn()
    else:
        retinanet.freeze_bn()

    print("Evaluating model...")
    results = csv_eval.evaluate(dataset_val, retinanet, iou_threshold=float(parser.iou_threshold))
    print(results)


if __name__ == '__main__':
    main()