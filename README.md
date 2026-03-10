# multimodal-emotion-recognition

This repository contains the implementation for the research paper:

**“Multi-Modal Fusion with Hierarchical Attention for Video Emotion Analysis.”**

The proposed system integrates visual, audio, and textual modalities using deep learning models and an attention-based fusion mechanism to improve emotion recognition performance from video data.

## Architecture Overview

The framework consists of three modality-specific branches:

* **Visual Branch:** Convolutional neural network for extracting spatial features from video frames.
* **Audio Branch:** CNN-based audio feature extraction from speech signals.
* **Text Branch:** BERT-based textual representation learning.
* **Fusion Module:** Hierarchical attention-based multimodal fusion.
* **Classifier:** Fully connected neural network for emotion classification.

## Dataset

The experiments are conducted using the **RAVDESS** dataset.

Dataset download link:
https://zenodo.org/record/1188976

Due to dataset licensing restrictions, the dataset is not included in this repository.

## Requirements

Python 3.10+

Main libraries:

* PyTorch
* Transformers
* Librosa
* OpenCV
* Torchvision
* NumPy
* Pandas

Install dependencies:

pip install -r requirements.txt

## Project Structure

src/

models/

visual_model.py
audio_model.py
text_model.py
multimodal_model.py

fusion/

attention_fusion.py

dataset_loader.py
train.py

## Training

Run the following command:

python train.py

## Model

The trained model weights are saved as:

models/multimodal_model.pth

## Citation

If you use this work, please cite the associated publication.
