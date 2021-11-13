import albumentations as A
import cv2


class AugmentationTransforms(object):
    @staticmethod
    def coco():
        transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=1024, interpolation=cv2.INTER_CUBIC, p=1.0),
                A.RandomCrop(height=512, width=512, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ToGray(p=0.2)
            ],
            bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3, label_fields=['class_labels'])
        )


def augment_coco(transform, image, boxes, class_labels):
    transformed = transform(image=image, boxes=boxes, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_boxes = transformed['boxes']
    transformed_class_labels = transformed['class_labels']
    return transformed_image, transformed_boxes, transformed_class_labels
