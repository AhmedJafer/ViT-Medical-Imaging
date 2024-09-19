
![VI](https://github.com/user-attachments/assets/f2be9774-137c-49dd-80cf-ce6d65dc5d49)

# Vision Transformer in the Medical Field

## Overview

This research project investigates the application of Vision Transformer (ViT) in radiographic diagnosis, focusing on:

1. Classification of lung cancer tumors using combined PET/CT modality
2. Classification of brain tumors from MRI

Our goals are to:

- Assist healthcare professionals in early diagnosis of these conditions
- Increase survival rates and improve overall patient well-being
- Compare the performance of CNN models with Transformer models
- Analyze attention maps for both ViT and CNN models
- Explore the impact and severity of data leakage in medical imaging

## Repository Structure

The repository contains two main directories, each representing one of the datasets used in our investigation:

1. Lung Cancer
2. Brain Tumor

## Lung Cancer Dataset

### Data Preparation

1. Download the Lung-PET-CT-Dx data from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/).

2. Organize patients by cancer type:
   ```
   python Preprocess.py <image_dir> <annotation_dir> <patient_id> <output_dir>
   ```
   - `<image_dir>`: Folder containing all patient folders
   - `<annotation_dir>`: Folder containing all annotation files
   - `<patient_id>`: Patient ID represented by one of the following letters: ["A", "B", "E", "G"]
   - `<output_dir>`: Directory to contain all processed patient folders

3. Remove PET and CT images, keeping only integrated PET-CT images:
   ```
   python PET_CT_Removal.py <main_folder>
   ```
   - `<main_folder>`: Directory containing all patient folders

### Modeling

Execute the following notebooks in order:

1. **Lung Cancer Data Preprocessing**
   - Exploratory Data Analysis (EDA)
   - BRISQUE deployment
   - Data splitting
   - Data augmentation

2. **Image Pre-Processing**
   - Applying smoothing
   - Filtering
   - Enhancing image and edges

3. **Models**
   - Training and testing of CNN-based and Transformer-based models
   - Evaluation of data leakage
   - Summarization of results

## Brain Tumor Dataset

Download the dataset from [Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427).

Execute the following notebooks in order:

1. **Pre-Processing**
   - Exploratory Data Analysis (EDA)
   - Data splitting
   - Data augmentation

2. **CNN Models**
   - Training and testing of CNN-based models

3. **ViT Models**
   - Training and testing of Transformer-based models

## Attention Map Visualization

The `Attention_Map_Visualization.ipynb` notebook includes the visualization of ResNet and ViT attention maps.

## Results

### Lung Cancer Classification

![Lung cancer results](https://github.com/user-attachments/assets/9f4b9896-67a0-424e-ab9c-ea2c0fc4c459)

### Brain Tumor Classification

![Brain Tumor result](https://github.com/user-attachments/assets/3c53ccaf-51c4-4bf7-a7ed-db2bee9263fb)

### Attention Map Visualization
<img src="https://github.com/user-attachments/assets/fbdd8a26-774d-473c-a01f-82b1ad3e3daa" width="50%" />

The figure represents the attention maps for
different models across various scenarios: A) Models trained on the brain tumor dataset.
B) Models trained on the lung cancer dataset. C) Models trained on the leaked lung
cancer dataset.


### Data Leakage Analysis

Our study on data leakage in the lung cancer dataset revealed significant insights:

1. **Methodology**: We introduced two types of data leakage during data splitting:
   - Splitting without considering patient boundaries, keeping original labels.
   - Same splitting method, but with randomly assigned labels.

2. **Model Performance**:
   - ResNet:
     - Achieved over 99% accuracy across all metrics in both approaches.
     - Showed high sensitivity to data leakage.
   - ViT B/16:
     - Improved to over 70% in most metrics with original labels.
     - Performed similar to non-leaked data with random labels, but still exceeded the 33.33% benchmark.

3. **Key Findings**:
   - Both models showed inflated performance due to data leakage.
   - ResNet appeared more susceptible to data leakage than ViT B/16.
   - ViT B/16's attention maps focused on the entire image rather than specific areas like tumors, indicating potential "memorization" of images.

4. **Implications**:
   - Results comparable to previous studies suggest possible widespread data leakage issues in the field.
   - Highlights the critical importance of careful data handling in medical imaging AI research.
   - Demonstrates the need for robust validation techniques to ensure model generalizability.

This analysis underscores the severity of data leakage in deep learning models for medical imaging and emphasizes the importance of rigorous data management practices in AI research.

## Model Weights

All model weights for both datasets can be downloaded from [Google Drive](https://drive.google.com/file/d/1g823_CNVHnPJF2k4lAM_uGuCf_XQXc_C/view?usp=sharing).

## Conclusion and Future Work

Our study on Vision Transformers (ViT) in radiographic diagnosis yielded several important insights:

- ViTs performed well in brain tumor detection but were slightly outperformed by CNNs.
- Both model types struggled with lung cancer detection, highlighting dataset complexity.
- Data leakage significantly impacted results, with CNNs showing higher sensitivity than ViTs.
- Attention map analysis revealed CNNs focus on specific regions, while ViTs capture broader features.

**Limitations** included small dataset sizes, limited GPU resources, and inconsistencies in previous research using the same dataset.

**Future research directions**
- Improving ViT interpretability through attention map analysis.
- Exploring Transformers in other medical imaging tasks (e.g., object detection, segmentation).
- Enhancing ViT performance on smaller datasets.
- Developing robust techniques to prevent and detect data leakage.

This study underscores the potential of Vision Transformers in medical image analysis while emphasizing the need for data integrity and proper experimental design in AI research.

## Requirements

All required libraries are listed in the respective Jupyter notebooks. Please refer to each notebook for the specific libraries needed to run that part of the project.

## Acknowledgements

This project was completed by Ahmed Jafar under the supervision of Prof. Luca Citi as part of a master's thesis.
