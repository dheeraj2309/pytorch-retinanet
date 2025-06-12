from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

import cv2
from PIL import Image

# NEW: Import the albumentations library
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        # MODIFIED: Albumentations works with uint8 images, so we don't divide by 255.0 here.
        # The Normalizer will handle the conversion to float and division.
        return img.astype(np.uint8) 

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        # MODIFIED: Albumentations works with uint8 images, so we don't divide by 255.0 here.
        # The Normalizer will handle the conversion to float and division.
        return img.astype(np.uint8)

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    
    # Find max height and width from the Tensors
    max_height = max(s.shape[1] for s in imgs)
    max_width = max(s.shape[2] for s in imgs)
    batch_size = len(imgs)

    # Create a padded image tensor
    padded_imgs = torch.zeros(batch_size, 3, max_height, max_width)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :, :img.shape[1], :img.shape[2]] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    # The image tensor is already in the correct (B, C, H, W) format
    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

class Resizer(object):
    """Resizes a numpy image and its annotations, and converts them to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots, scale = sample['img'], sample['annot'], sample.get('scale', 1.0)

        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        scale_factor = min_side / smallest_side
        largest_side = max(rows, cols)

        if largest_side * scale_factor > max_side:
            scale_factor = max_side / largest_side
        
        # Resize using skimage, keeping it as a numpy array for now
        image_resized = skimage.transform.resize(
            image, 
            (int(round(rows * scale_factor)), int(round(cols * scale_factor))),
            preserve_range=True, # Important: keeps the 0-255 range
            mode='constant'
        ).astype(np.float32)

        rows, cols, cns = image_resized.shape
        pad_w = 32 - rows % 32 if rows % 32 != 0 else 0
        pad_h = 32 - cols % 32 if cols % 32 != 0 else 0

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns), dtype=np.float32)
        new_image[:rows, :cols, :] = image_resized

        annots[:, :4] *= scale_factor

        # Convert to Tensors at the end of this step
        return {
            'img': torch.from_numpy(new_image), 
            'annot': torch.from_numpy(annots), 
            'scale': scale_factor
        }


# MODIFIED: Replaced the old Augmenter with a new powerful one using Albumentations
class Augmenter(object):
    """
    Apply a series of augmentations from the Albumentations library.
    This handles both image and bounding box transformations.
    """
    def __init__(self):
        # Define a composition of augmentations.
        # p=0.5 means a 50% chance of applying the transformation.
        self.transform = A.Compose([
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),

            # Color/photometric augmentations
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.RGBShift(p=0.5),
            
            # Blur and noise
            A.Blur(blur_limit=3, p=0.3),
            A.GaussNoise(p=0.3),

            # Occlusion
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])) # 'pascal_voc' is [x_min, y_min, x_max, y_max]

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        if annots.shape[0] == 0:
            # No annotations to transform, just return the image
            return {'img': image, 'annot': annots}

        # Albumentations requires bboxes and class labels as separate arguments
        bboxes = annots[:, :4]
        class_labels = annots[:, 4]

        try:
            # Apply the transformations
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            transformed_image = transformed['image']
            transformed_bboxes = np.array(transformed['bboxes'])
            transformed_class_labels = np.array(transformed['class_labels'])

            if len(transformed_bboxes) == 0:
                # If all bboxes were removed by the augmentation, return an empty annotation
                transformed_annots = np.zeros((0, 5), dtype=np.float32)
            else:
                # Recombine the transformed bboxes and labels into the annotation format
                transformed_annots = np.hstack((transformed_bboxes, transformed_class_labels.reshape(-1, 1)))
            
            return {'img': transformed_image, 'annot': transformed_annots}

        except Exception as e:
            print(f"Could not apply augmentations: {e}")
            # If an error occurs, return the original sample
            return {'img': image, 'annot': annots}


class Normalizer(object):
    """Normalizes a Tensor image."""
    def __init__(self):
        # Define mean and std as Tensors, not numpy arrays
        # Use .view to make them broadcastable for (C, H, W) images
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, sample):
        # The sample now contains Tensors
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        # Ensure mean and std are on the same device as the image
        # This is important for GPU usage
        mean = self.mean.to(image.device)
        std = self.std.to(image.device)

        # Permute the image from (H, W, C) to (C, H, W) for PyTorch operations
        image = image.permute(2, 0, 1)

        # Perform normalization using PyTorch operations
        # The image is already float32 from the Resizer
        normalized_image = (image / 255.0 - mean) / std
        
        # The collater will handle the final batching
        return {'img': normalized_image, 'annot': annots, 'scale': scale}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]