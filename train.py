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
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
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
    
    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []
        
        # Training loop
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
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
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)), end='\r')
                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        # Validation and model saving
        current_metric = -1.0
        if parser.dataset == 'coco' and dataset_val is not None:
            print('\nEvaluating dataset')
            eval_results = coco_eval.evaluate_coco(dataset_val, retinanet)
            current_metric = eval_results['map']
        elif parser.dataset == 'csv' and dataset_val is not None:
            print('\nEvaluating dataset')
            eval_results = csv_eval.evaluate(dataset_val, retinanet)
            # Assuming csv_eval also returns a dict like {'map': ...}
            current_metric = eval_results['map']
        
        # Pass the mean epoch loss to the scheduler
        scheduler.step(np.mean(epoch_loss))

        # Save a checkpoint of the current epoch
        epoch_checkpoint_path = os.path.join(parser.checkpoint_path, '{}_{}_retinanet_{}.pt'.format(parser.dataset, parser.backbone, epoch_num))
        torch.save(retinanet.module.state_dict(), epoch_checkpoint_path)

        # Early stopping and best model logic
        if current_metric > best_metric:
            best_metric = current_metric
            epochs_no_improve = 0
            best_model_path = os.path.join(parser.checkpoint_path, 'best_model.pt')
            torch.save(retinanet.module.state_dict(), best_model_path)
            print(f"Validation metric improved to {best_metric:.4f}. Saved best model to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"Validation metric did not improve. Current: {current_metric:.4f}, Best: {best_metric:.4f}. Count: {epochs_no_improve}/{parser.early_stopping_patience}")
            if epochs_no_improve >= parser.early_stopping_patience:
                print("Early stopping triggered. Training finished.")
                break

    # Final save of the model state (optional, as best model is already saved)
    final_model_path = os.path.join(parser.checkpoint_path, 'model_final.pt')
    retinanet.eval()
    torch.save(retinanet.module.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")


if __name__ == '__main__':
    main()