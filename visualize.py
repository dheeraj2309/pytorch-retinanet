import numpy as np
import torchvision
import time
import os
import argparse
import cv2
import torch
import skimage.transform # We need this for resizing
from IPython.display import display, Image

# Import necessary classes from your project structure
from retinanet import model
from retinanet.dataloader import CSVDataset, UnNormalizer

# To avoid duplicating code and ensure consistency, we create a Preprocess class
# for visualization that mirrors the Resizer -> Normalizer pipeline from your training validation set.
class Preprocess(object):
    """
    A self-contained class for validation/inference pre-processing.
    It mimics the Resizer -> Normalizer chain to ensure data is handled identically.
    """
    def __init__(self, min_side=608, max_side=1024):
        self.min_side = min_side
        self.max_side = max_side
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, sample):
        # 1. Unpack initial sample {'img': np.array, 'annot': np.array}
        image, annots = sample['img'], sample['annot']

        # --- Resizer Logic ---
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
        
        image_tensor = torch.from_numpy(new_image_padded)
        annot_tensor = torch.from_numpy(annots)

        # --- Normalizer Logic ---
        image_tensor = image_tensor.permute(2, 0, 1) # Permute to (C, H, W)
        normalized_image = (image_tensor / 255.0 - self.mean) / self.std

        # 4. Return the final dictionary, just like the training pipeline
        return {
            'img': normalized_image,
            'annot': annot_tensor,
            'scale': scale_factor
        }

def visualize(args):
    # --- 1. SETUP ---
    print('Loading dataset...')
    # Use our new Preprocess class for consistent data handling
    dataset_val = CSVDataset(
        train_file=args.csv_val, 
        class_list=args.csv_classes, 
        transform=Preprocess()
    )

    print('Loading model...')
    # Create an instance of your model architecture
    retinanet = model.efficientnet_b0_retinanet(num_classes=dataset_val.num_classes())
    
    # Load the saved weights (state_dict) into the model
    # Use map_location to ensure it loads correctly whether on CPU or GPU
    state_dict = torch.load(args.model, map_location=torch.device('cpu'))
    retinanet.load_state_dict(state_dict)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    retinanet = retinanet.to(device)
    retinanet.eval()
    print(f'Model and dataset loaded. Using device: {device}')

    # --- 2. VISUALIZATION ---
    unnormalize = UnNormalizer()
    
    # Generate a unique color for each class
    import matplotlib.pyplot as plt
    colors = plt.cm.get_cmap('hsv', dataset_val.num_classes()).colors
    colors = (np.array(colors) * 255).astype(np.uint8)

    for i in range(args.num_images):
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
            img = cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Draw ground truth boxes in GREEN
            if data['annot'].shape[0] > 0:
                print('Ground Truth Boxes:')
                for gt_box in data['annot']:
                    x1_gt, y1_gt, x2_gt, y2_gt, class_id_gt = gt_box.int().numpy()
                    class_name_gt = dataset_val.labels[class_id_gt]
                    print(f'  - {class_name_gt}')
                    cv2.rectangle(img, (x1_gt, y1_gt), (x2_gt, y2_gt), color=(0, 255, 0), thickness=2)

            print('Detected Boxes:')
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
            
            # Display using IPython.display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, img_encoded = cv2.imencode('.png', img_rgb)
            display(Image(data=img_encoded.tobytes()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for visualizing RetinaNet detections.')
    parser.add_argument('--csv_classes', help='Path to file containing class list')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations')
    parser.add_argument('--model', help='Path to model state_dict (.pt) file')
    parser.add_argument('--num_images', help='Number of images to visualize', type=int, default=10)
    parser.add_argument('--score_threshold', help='Confidence score threshold for detections', type=float, default=0.5)
    args = parser.parse_args()
    visualize(args)