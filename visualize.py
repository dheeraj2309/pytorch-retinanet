import numpy as np
import torchvision
import time
import os
import copy
import argparse
import cv2
import torch
# import matplotlib.pyplot as plt # We don't need matplotlib for display anymore, but keep for colors

# NEW: Import the necessary IPython display functions
from IPython.display import display, Image

# Import the necessary classes from your project structure
# In a Kaggle notebook, you would have cloned your repo first
from retinanet import model
from retinanet.dataloader import CSVDataset, Preprocess, UnNormalizer

def visualize(args):
    # --- 1. SETUP ---
    print('Loading dataset...')
    # Use the robust Preprocess class for consistent data handling
    dataset_val = CSVDataset(
        train_file=args.csv_val, 
        class_list=args.csv_classes, 
        transform=Preprocess()
    )

    print('Loading model...')
    # Create an instance of your model architecture
    # Make sure the backbone and num_classes match the saved model
    retinanet = model.efficientnet_b0_retinanet(num_classes=dataset_val.num_classes())
    
    # Load the saved weights (state_dict) into the model
    state_dict = torch.load(args.model, map_location=torch.device('cpu'))
    retinanet.load_state_dict(state_dict)
    
    # Set up device
    use_gpu = True
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        retinanet = retinanet.to(device)
    else:
        device = torch.device('cpu')

    retinanet.eval()
    print('Model and dataset loaded.')

    # --- 2. VISUALIZATION ---
    unnormalize = UnNormalizer()
    
    # Define colors for bounding boxes
    # We can still use matplotlib's colormap functionality to generate colors
    import matplotlib.pyplot as plt
    colors = plt.cm.get_cmap('hsv', dataset_val.num_classes()).colors
    colors = (np.array(colors) * 255).astype(np.uint8)

    # Loop for a specific number of images
    for i in range(args.num_images):
        # Get a sample from the validation set
        data = dataset_val[i]
        
        detection_count = 0

        with torch.no_grad():
            st = time.time()
            
            # Get model output
            scores, labels, boxes = retinanet(data['img'].to(device).float().unsqueeze(dim=0))
            
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            print(f'\n--- Image {i+1}/{args.num_images} ---')
            print(f'Elapsed time: {time.time()-st:.4f}s')

            confident_indices = np.where(scores > args.score_threshold)[0]
            
            # Un-normalize the image for display
            img = data['img'].cpu()
            img = unnormalize(img)
            img = img.permute(1, 2, 0).numpy() # Permute to (H, W, C)
            
            # Convert to a format that OpenCV can use for drawing (uint8 BGR)
            # OpenCV works with BGR, but we need RGB for final display. We'll convert back later.
            img = cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Draw ground truth boxes (optional)
            if data['annot'].shape[0] > 0:
                print('Ground Truth Boxes:')
                for gt_box in data['annot']:
                    x1_gt, y1_gt, x2_gt, y2_gt, class_id_gt = gt_box.int().numpy()
                    class_name_gt = dataset_val.labels[class_id_gt]
                    print(f'  - {class_name_gt}')
                    cv2.rectangle(img, (x1_gt, y1_gt), (x2_gt, y2_gt), color=(0, 255, 0), thickness=2)

            print('Detected Boxes:')
            # Draw detected boxes
            for j in confident_indices:
                detection_count += 1
                box = boxes[j, :]
                label_id = labels[j]
                score = scores[j]
                
                class_name = dataset_val.labels[label_id]
                color = colors[label_id].tolist()
                
                caption = f'{class_name} {score:.2f}'
                print(f'  - {caption} at {box.astype(int)}')

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
                
                (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                cv2.putText(img, caption, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            print(f'Total detections on this image: {detection_count}')
            
            # --- NEW: Display using IPython.display ---
            # 1. Convert the final image (which is in BGR format from OpenCV) back to RGB.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 2. Encode the image to a format that can be displayed (like PNG).
            _, img_encoded = cv2.imencode('.png', img_rgb)
            
            # 3. Display the image directly in the notebook output.
            display(Image(data=img_encoded.tobytes()))


# This is how you would run it in a script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for visualizing RetinaNet detections.')
    
    parser.add_argument('--csv_classes', help='Path to file containing class list')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations')
    parser.add_argument('--model', help='Path to model state_dict (.pt) file')
    
    parser.add_argument('--num_images', help='Number of images to visualize', type=int, default=10)
    parser.add_argument('--score_threshold', help='Confidence score threshold for detections', type=float, default=0.5)

    args = parser.parse_args()
    visualize(args)