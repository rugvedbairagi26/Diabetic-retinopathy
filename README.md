🩺 Diabetic Retinopathy Severity Classification using Deep Learning

This project focuses on automatic classification of diabetic retinopathy (DR) severity levels from retinal fundus images using convolutional neural networks and transfer learning. The goal is to build a reliable multi-class classifier that can support early screening and grading of diabetic retinopathy.

🎯 Problem Statement

Diabetic Retinopathy is a diabetes-related eye disease that damages retinal blood vessels and can lead to blindness if not detected early. Manual grading of retinal images is time-consuming and requires trained ophthalmologists.

This project aims to:

classify retinal fundus images into 5 severity levels

evaluate performance on imbalanced clinical data

analyze model behavior across minority disease classes

📊 Dataset

Dataset: APTOS 2019 Blindness Detection (Kaggle)

Images: ~3,600 labeled retinal fundus images

Classes:

0 – No DR

1 – Mild

2 – Moderate

3 – Severe

4 – Proliferative DR

Challenges

Strong class imbalance, especially in Severe and Proliferative classes

Large variations in brightness, contrast, and image quality

🔍 Exploratory Data Analysis

Before training, I analyzed:

✔ Class Distribution

The dataset is highly imbalanced, with far fewer samples in advanced disease categories.

✔ Visual Inspection

Random samples from each class were visualized to inspect:

lesion patterns

illumination differences

background artifacts

This analysis motivated careful preprocessing and conservative augmentation.

🔀 Dataset Splitting Strategy

To ensure fair evaluation:

Training set: 70%

Validation set: 15%

Test set: 15%

✔ Stratified Splitting

Splits were created using stratified sampling so that each subset preserves original class proportions.
This avoids biased evaluation on rare classes.

All splits were saved as CSV files for reproducibility.

⚙️ Preprocessing and Data Loading

Image resizing to 224 × 224

Model-specific normalization:

ResNet preprocessing for ResNet model

EfficientNet preprocessing for EfficientNet model

Horizontal flipping for augmentation

To improve I/O speed, images were copied from Google Drive to local Colab storage before training.

🧠 Models and Training
🔹 Baseline Model — ResNet50

Pretrained on ImageNet

Backbone frozen

Custom classifier head:

Global Average Pooling

Dense (256)

Dropout (0.5)

Softmax (5 classes)

Purpose: evaluate how generic visual features perform on retinal images.

🔹 Improved Model — EfficientNetB0

EfficientNet was chosen due to:

better parameter efficiency

stronger feature extraction

Stage 1: Feature Extraction

Backbone frozen

Only classifier trained

Stage 2: Fine-Tuning

Upper convolution layers unfrozen

Lower layers kept frozen

Reduced learning rate

This allows adaptation to retinal lesion patterns while preserving general visual features.

📈 Evaluation Results
🔹 Baseline (ResNet50)

Accuracy: ~78%

Very low recall for:

Severe DR

Proliferative DR

This indicates strong bias toward majority classes due to imbalance.

🔹 Improved Model (EfficientNetB0 + Fine-Tuning)

Accuracy: ~83%

Improved recall for minority classes

Better macro-averaged F1 score

Remaining confusion mainly occurs between:

Moderate vs Severe DR

Severe vs Proliferative DR

These categories also show overlap in clinical grading.

🧠 Analysis and Limitations

Severe class imbalance still limits recall for advanced disease stages

Borderline clinical cases are difficult even for human experts

Model trained on single dataset — generalization to other hospitals not guaranteed

These limitations motivate further research into:

class-balanced loss functions

multi-dataset training

lesion-aware attention mechanisms

💾 Model Saving

Trained models are saved for:

inference experiments

Grad-CAM visualization

further fine-tuning

Both baseline and final models are stored for future comparison.

🚀 Future Work

Planned extensions:

✔ Explainability

Grad-CAM to verify that predictions focus on lesion regions instead of background artifacts

✔ Class Imbalance Handling

Focal loss

Class-weighted loss functions

✔ Generalization

Evaluation on external datasets (IDRiD, Messidor)

✔ Clinical Relevance

Binary DR screening → severity grading pipeline

🧪 Environment

TensorFlow / Keras

Training performed using Kaggle / Colab GPU environment

Image loading via Keras ImageDataGenerator

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
