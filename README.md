# Diabetic-retinopathy

🩺 Diabetic Retinopathy Severity Classification using CNNs
📌 Project Motivation

Diabetic Retinopathy (DR) is a common complication of diabetes and can lead to permanent vision loss if not detected in time. Manual screening of retinal images is time-consuming and requires trained ophthalmologists, so automated screening using deep learning can be useful as a support tool.

In this project, I worked on multi-class classification of diabetic retinopathy severity using retinal fundus images and transfer learning with convolutional neural networks. The goal was not only to train a model, but also to follow a proper ML workflow similar to research projects.

📊 Dataset

Dataset: APTOS 2019 Blindness Detection (Kaggle)

Each image is labeled into one of five classes:

0 → No DR

1 → Mild

2 → Moderate

3 → Severe

4 → Proliferative DR

Observations from the Dataset

The dataset is highly imbalanced, with very few images in Severe and Proliferative classes.

Images vary a lot in:

brightness

sharpness

background noise

Resolution also varies, so resizing strategy becomes important.

Because of these factors, classification is not trivial even for deep models.

🔍 Exploratory Data Analysis

Before training, I performed basic EDA:

Checked class distribution → confirmed strong imbalance.

Visualized random samples from each class to understand lesion patterns.

Verified that image paths are correct and readable.

This step helped in understanding why simple accuracy alone is not enough to evaluate medical models.

✂️ Data Splitting Strategy

To avoid biased evaluation, I used stratified splitting:

Train: 70%

Validation: 15%

Test: 15%

Stratification was done using class labels so that each split has similar class distribution.

The splits were saved as CSV files to ensure reproducibility.

⚙️ Data Pipeline Optimization

Initially, loading images directly from Google Drive caused very slow training due to I/O bottlenecks.
To fix this:

All images were copied to Colab local storage (/content/train_images)

Image paths in CSV files were updated accordingly

This significantly improved training speed and GPU utilization.

🧠 Models Used

I trained two different CNN architectures using transfer learning:

🔹 Baseline Model — ResNet50

Pretrained on ImageNet

Backbone kept frozen

Custom classifier added:

Global Average Pooling

Dense layer (256 units)

Dropout (0.5)

Softmax output for 5 classes

Training:

Optimizer: Adam (lr = 1e-3)

Loss: Sparse categorical cross-entropy

Epochs: 8

This model acts as a baseline for comparison.

🔹 Improved Model — EfficientNetB0

EfficientNet was selected because it provides better performance with fewer parameters.

Phase 1 — Feature Extraction

Backbone frozen

Only classifier trained

Epochs: 8

Learning rate: 1e-3

Phase 2 — Fine-Tuning

Upper layers of backbone unfrozen

Lower layers kept frozen to avoid overfitting

Learning rate reduced to 1e-4

Epochs: 15

Fine-tuning helped the model adapt pretrained features to retinal lesion patterns.

📈 Evaluation Metrics

Models were evaluated on the held-out test set using:

Accuracy

Precision, Recall, F1-score (per class)

Confusion Matrix

🔹 ResNet50 Results

Good performance on majority classes (No DR, Moderate)

Very low recall for Severe and Proliferative classes

This shows the effect of class imbalance and why accuracy alone is misleading.

🔹 EfficientNet Results

Higher overall accuracy

Better recall for minority classes compared to ResNet

More balanced macro-averaged F1 score

This confirms that both architecture choice and fine-tuning improve medical image classification performance.

🔍 Error Analysis

Wrong predictions were visualized manually to understand failure cases.

Common error patterns:

Confusion between Moderate and Severe DR

Confusion between Severe and Proliferative DR

Low contrast images where lesions are not clearly visible

This highlights that even strong CNNs struggle with subtle medical differences and that interpretability is important.

💾 Saved Models

Final trained models were saved for future experiments:

resnet50_baseline.h5

efficientnet_final.h5

🔬 What I Learned from This Project

How to properly handle class imbalance in medical datasets

Why stratified splitting is important

Difference between feature extraction and fine-tuning

Importance of class-wise metrics instead of only accuracy

Practical issues like data loading bottlenecks in Colab

This project helped me understand how real ML pipelines are built beyond just fitting a model.

🚀 Future Work

The following improvements are planned:

🔹 Grad-CAM Visualization

I plan to apply Grad-CAM to visualize where the CNN focuses in retinal images, to check if predictions are based on lesion areas instead of background artifacts. This is important for explainable medical AI.

🔹 Imbalance Handling

Future experiments will include:

class-weighted loss

focal loss

to improve detection of severe DR stages.

🔹 Higher Resolution Training

Since retinal lesions are small, future work will try training with higher resolution inputs or patch-based approaches.

🛠 Tools and Libraries

Python

TensorFlow / Keras

NumPy, Pandas

OpenCV

Scikit-learn

Google Colab (GPU)

👤 About Me

I am a second-year Computer Science student interested in:

Computer Vision

Medical Imaging

Deep Learning

Robotics and AI

This project is part of my preparation for applying to research internships in AI and medical imaging.
