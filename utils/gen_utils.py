import cv2
import tensorflow_datasets as tfds


def read_image(image_filename):
    image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    return image


def download_coco_tfrecords(coco_version, data_dir):
    if coco_version not in ['2014', '2017']:
        raise ValueError('The argument coco_version must be one of "2014" and "2017".')

    coco_name = 'coco/{}'.format(coco_version)

    ds = tfds.load(coco_name, split='all', with_info=True, data_dir=data_dir)
    return None
