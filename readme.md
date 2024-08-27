# Vision Transformer in the Medical Field

## Overview
In this research, we will investigate the utilization of Vision Transformer (ViT) in
radiographic diagnosis, focusing on the classification of lung cancer tumors using the
combined PET/CT modality, as well as brain tumors from MRI. This study aims to
assist healthcare professionals in the early diagnosis of these conditions, thus increasing
survival rates and elevating overall well-being. In addition, we will compare the
performance of CNN models with Transformer models and analyze the attention maps
for both ViT and CNN models, gaining insights into their behavior and interpretability.
Finally, we will explore the impact and severity of data leakage in medical imaging and
examine its effect on the current literature on the topic.

## Repository Overview
This repository contains all the code and files necessary to complete this project. 
The repository is composed of two directories, with each directory representing one of the 
datasets utilized in our investigation.

# Lung Cancer 
In this section, we outline the steps required to replicate our findings and run our code. Before proceeding,
please download the Lung-PET-CT-Dx data from the following link:
[Lung-PET-CT-Dx Data](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/).

## Data Preparation
After downloading the data,  This dataset comprises data from 355 patients with
their names/IDs indicating their diagnosis. Specifically, patients marked with the letter
’A’ have been diagnosed with Adenocarcinoma, ’B’ have Small Cell Carcinoma, ’E’
represents Large Cell Carcinoma and ’G’ have Squamous Cell Carcinoma.


**The first step** involves selecting all patients diagnosed with Adenocarcinoma (marked with 'A') and organizing their data into a separate folder. 
Once this is done, run the following code:

``python Proprocess.py <image_dir> <annotation_dir> <patient_id> <output_dir> ``

`<image_dir>:` The folder that contain all patients folders 

`<annotation_dir>:` The folder that contain all annoataions files 

`<patient_id>:` Patient ID represented by one of the following letters: `["A","B","E","G"] `

`<output_dir>:` Directory the contain all patients folders 

Repeat this process for each cancer type. After processing, all the patient data will be consolidated into 
one `output_dir`

**Secondly**, execute the following command to remove all PET and CT images, retaining only the integrated PET-CT images:

``python PET_CT_Removal.py <main_folder>``

where 
``<main_folder>:`` is the directory the contain all patients folders 

## Modelling

**The following Notebooks should be executed in the specified sequence:**

1. Lung Cancer Data Preprocessing

This notebook includes:

- Exploratory Data Analysis (EDA)
- BRISQUE deployment
- Data splitting
- Data augmentation


2. Image Pre-Processing

This step involves:
- Applying smoothing
- Filtering
- Enhancing image and edges

3. Models

This notebook includes:

- Training and testing of CNN-based and Transformer-based models
- Evaluation of data leakage
- Summarization of results

# Brain Tumor:

For the brain tumor dataset, it is more organized and requires less pre-processing.
you can download the dataset from [here](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) 

**The following Notebooks should be executed in the specified sequence:**

1. Pre-Processing 

This notebook includes:

- Exploratory Data Analysis (EDA)
- Data splitting
- Data augmentation

2. CNN Models

Training and testing of CNN-based models 

3. ViT models 

Training and testing of Transformer-based models

4. Results summarization


# Attention Map Visualization

The Notebook includes the visualization of ResNet and ViT attention maps


# Note 

All model weights for both datasets can be downloaded from here
[here](https://essexuniversity.box.com/s/w7d2ueen596lk2ldw0lbmb5nbt8tmjtw)