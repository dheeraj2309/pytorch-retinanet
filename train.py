import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader
from retinanet import coco_eval
from retinanet import csv_eval
import os # NEW: import os for path joining
import csv

# assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--backbone', help='Backbone network, must be one of resnet18, 34, 50, 101, 152 or efficientnet-b0', type=str, default='resnet50')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser.add_argument('--freeze_backbone', help='Freeze backbone weights', action='store_true')
    parser.add_argument('--freeze_fpn', help='Freeze FPN weights', action='store_true')
    parser.add_argument('--freeze_regression_head', help='Freeze regression head weights', action='store_true')
    parser.add_argument('--freeze_classification_head', help='Freeze classification head weights', action='store_true')
    
    # NEW: Add arguments for early stopping and model saving
    parser.add_argument('--early_stopping_patience', help='Number of epochs to wait for improvement before stopping.', type=int, default=10)
    parser.add_argument('--checkpoint_path', help='Path to save model checkpoints.', type=str, default='checkpoints')


    parser = parser.parse_args(args)
    
    # NEW: Create checkpoint directory if it doesn't exist
    if not os.path.exists(parser.checkpoint_path):
        os.makedirs(parser.checkpoint_path)

    # Create the data loaders
    if parser.dataset == 'coco':
        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Augmenter(), Resizer(), Normalizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on csv,')
        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on csv,')
        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Augmenter(), Resizer(), Normalizer()]))
        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Resizer(), Normalizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    freeze_args = {
        'freeze_backbone': parser.freeze_backbone,
        'freeze_fpn': parser.freeze_fpn,
        'freeze_regression_head': parser.freeze_regression_head,
        'freeze_classification_head': parser.freeze_classification_head
    }

    if parser.backbone == 'efficientnet-b0':
        retinanet = model.efficientnet_b0_retinanet(num_classes=dataset_train.num_classes(), pretrained=True, **freeze_args)
    elif 'resnet' in parser.backbone:
        resnet_depth = int(parser.backbone.replace('resnet', ''))
        factory = {18: model.resnet18, 34: model.resnet34, 50: model.resnet50, 101: model.resnet101, 152: model.resnet152}
        retinanet = factory[resnet_depth](num_classes=dataset_train.num_classes(), pretrained=True, **freeze_args)
    else:
        raise ValueError('Unsupported backbone.')

    use_gpu = True
    if use_gpu and torch.cuda.is_available():
        retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, retinanet.parameters()), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    
    # NEW: Initialize variables for early stopping and best model saving
    best_metric = -1.0
    epochs_no_improve = 0

    log_file_path = os.path.join(parser.checkpoint_path, 'training_log.csv')
    log_file = open(log_file_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    # Write header to the log file
    log_writer.writerow(['epoch', 'training_loss', 'classification_loss', 'regression_loss', 'validation_map', 'learning_rate'])
    
    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        epoch_cls_loss = []
        epoch_reg_loss = []
        
        # Training loop
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()

                # Store losses for epoch average
                epoch_loss.append(float(loss))
                epoch_cls_loss.append(float(classification_loss))
                epoch_reg_loss.append(float(regression_loss))

                # Print running loss for the current iteration
                print(
                    'Epoch: {} | Iteration: {} | Cls Loss: {:1.5f} | Reg Loss: {:1.5f} | Total Loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), float(loss)), end='\r')
                
            except Exception as e:
                print(e)
                continue

        # --- End of Epoch: Validation and Logging ---
        avg_train_loss = np.mean(epoch_loss)
        avg_cls_loss = np.mean(epoch_cls_loss)
        avg_reg_loss = np.mean(epoch_reg_loss)
        
        # Get current learning rate from optimizer
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n--- End of Epoch {epoch_num} ---")
        print(f"Average Training Loss: {avg_train_loss:.5f} | Cls: {avg_cls_loss:.5f} | Reg: {avg_reg_loss:.5f}")
        print(f"Current Learning Rate: {current_lr}")

        # Validation
        validation_map = -1.0
        if dataset_val is not None:
            print('Evaluating dataset...')
            if parser.dataset == 'coco':
                eval_results = coco_eval.evaluate_coco(dataset_val, retinanet)
            else: # 'csv'
                eval_results = csv_eval.evaluate(dataset_val, retinanet)
            validation_map = eval_results['map']
        
        # NEW: Write logs for the completed epoch
        log_writer.writerow([epoch_num, avg_train_loss, avg_cls_loss, avg_reg_loss, validation_map, current_lr])
        log_file.flush() # Ensure data is written to the file immediately

        scheduler.step(avg_train_loss)

        # Early stopping and best model saving
        if validation_map > best_metric:
            best_metric = validation_map
            epochs_no_improve = 0
            best_model_path = os.path.join(parser.checkpoint_path, 'best_model.pt')
            torch.save(retinanet.module.state_dict(), best_model_path)
            print(f"Validation mAP improved to {best_metric:.4f}. Saved best model.")
        else:
            epochs_no_improve += 1
            print(f"Validation mAP did not improve. Current: {validation_map:.4f}, Best: {best_metric:.4f}. Count: {epochs_no_improve}/{parser.early_stopping_patience}")
            if epochs_no_improve >= parser.early_stopping_patience:
                print("--- Early stopping triggered. ---")
                break
    
    # NEW: Close the log file at the end of training
    log_file.close()
    print("Training finished. Log file saved at:", log_file_path)
    # Final save of the model state (optional, as best model is already saved)
    final_model_path = os.path.join(parser.checkpoint_path, 'model_final.pt')
    retinanet.eval()
    torch.save(retinanet.module.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")


if __name__ == '__main__':
    main()