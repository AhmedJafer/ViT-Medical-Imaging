import os
import cv2
import numpy as np
import argparse


def channels_identical(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False
    return np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 1], image[:, :, 2])


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def process_folder(main_folder):
    for root, _, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith(('.png')):
                image_path = os.path.join(root, file)
                annotation_path = os.path.splitext(image_path)[0] + '.txt'

                if channels_identical(image_path):
                    remove_file(image_path)
                    remove_file(annotation_path)


def main():
    parser = argparse.ArgumentParser(description='Remove PET and CT')
    parser.add_argument('main_folder', type=str,
                        help='Path to the main folder containing subfolders with images and annotation files')
    args = parser.parse_args()

    process_folder(args.main_folder)


if __name__ == "__main__":
    main()