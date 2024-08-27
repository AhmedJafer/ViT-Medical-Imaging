import os
import shutil
import matplotlib.pyplot as plt
import pydicom
import xml.etree.ElementTree as ET
import argparse
from getUID import *
from get_data_from_XML import *

def dicom_to_png(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                dicom_data = pydicom.dcmread(dicom_path)
                png_data = dicom_data.pixel_array
                plt.imshow(png_data, cmap=plt.cm.gray)
                plt.axis('off')
                png_file = os.path.splitext(file)[0] + ".png"
                png_path = os.path.join(output_dir, png_file)
                plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
                plt.close()

def paths(image_directory, annotation_directory):
    images = os.listdir(image_directory)
    annotations = os.listdir(annotation_directory)
    pairs = [(os.path.join(image_directory, image_file), os.path.join(annotation_directory, ann_file))
             for image_file in images for ann_file in annotations if image_file[-5:] == ann_file]
    return pairs

def copy_image_folder(main_folder, filename):
    folder_path = os.path.join(main_folder, filename)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def remove_files_with_extension(folder_path, extension="dcm"):
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith(extension):
            os.remove(os.path.join(folder_path, file))

def annotation_saving(Coordinates, file_name, directory_path):
    x_tl, y_tl, x_br, y_br = Coordinates[0][:4]
    X_Centre = (x_tl + x_br) / 2 / 512
    Y_Centre = (y_tl + y_br) / 2 / 512
    width = (x_br - x_tl) / 512
    height = (y_br - y_tl) / 512
    annotation_format = f"{np.argmax(Coordinates[0][-4:]) + 1} {X_Centre} {Y_Centre} {width} {height}"
    annotation_name = os.path.splitext(file_name)[0] + ".txt"
    file_path = os.path.join(directory_path, annotation_name)
    with open(file_path, "w") as file:
        file.write(annotation_format)

def XML_Fixer(path, patient_id):
    for filename in os.listdir(path):
        if filename.endswith('.xml'):
            file_path = os.path.join(path, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()
            for name in root.iter('name'):
                if name.text != patient_id:
                    name.text = patient_id
            tree.write(file_path, encoding='utf-8', xml_declaration=True)

def main(image_dir, annotation_dir, patient_id, output_dir):
    Path = paths(image_dir, annotation_dir)
    for images_filename, annotation_filename in Path:
        dest_file_path = copy_image_folder(output_dir, images_filename[-5:])
        Dict = getUID_path(images_filename)
        XML_Fixer(annotation_filename, patient_id)
        annotations = XML_preprocessor(annotation_filename, num_classes=4).data
        checker = set()
        c = 0
        for k, v in annotations.items():
            try:
                dcm_path, dcm_name = Dict[k[:-4]]
                if dcm_path not in checker:
                    checker.add(dcm_path)
                    file_name = os.path.basename(dcm_path)
                    base, extension = os.path.splitext(file_name)
                    new_file_name = f"{base}_{c}{extension}"
                    adjusted_dest_file_path = os.path.join(dest_file_path, new_file_name)
                    c += 1
                    shutil.copy2(dcm_path, adjusted_dest_file_path)
                    annotation_saving(v, new_file_name, dest_file_path)
            except KeyError:
                pass
        dicom_to_png(dest_file_path, dest_file_path)
        remove_files_with_extension(dest_file_path, "dcm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DICOM and annotation files.")
    parser.add_argument("image_dir", type=str, help="Directory containing DICOM images")
    parser.add_argument("annotation_dir", type=str, help="Directory containing annotation files")
    parser.add_argument("patient_id", type=str, choices=["A", "B", "E", "G"], help="Patient ID to be set in XML files")
    parser.add_argument("output_dir", type=str, help="Directory to save the processed files")
    args = parser.parse_args()
    main(args.image_dir, args.annotation_dir, args.patient_id, args.output_dir)
