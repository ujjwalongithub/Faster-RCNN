import albumentations as A
import cv2
import numpy as np

COCO_TRANSFORM_TRAIN = A.Compose(
    [
        A.SmallestMaxSize(max_size=1024, interpolation=cv2.INTER_CUBIC, p=1.0),
        A.RandomCrop(height=512, width=512, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ToGray(p=0.2),
        A.Normalize(p=1.0)
    ],
    bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3, label_fields=['class_labels'])
)

COCO_TRANSFORM_TEST = A.Compose(
    [
        A.RandomCrop(height=512, width=512, p=1.0),
        A.Normalize(p=1.0)
    ],
    bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3, label_fields=['class_labels'])
)

def augment_coco_train(image, boxes, class_labels):
    boxes = boxes[:, [1,0,3,2]]
    transformed = COCO_TRANSFORM_TRAIN(image=image, bboxes=boxes, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_boxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']
    return transformed_image, transformed_boxes, transformed_class_labels


def augment_coco_test(image, boxes, class_labels):
    boxes = boxes[:, [1,0,3,2]]
    transformed = COCO_TRANSFORM_TEST(image=image, bboxes=boxes, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_boxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']
    return transformed_image, transformed_boxes, transformed_class_labels