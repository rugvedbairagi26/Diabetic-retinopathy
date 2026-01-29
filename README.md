🩺 Diabetic Retinopathy Severity Classification using Deep Learning

This project focuses on automatic classification of diabetic retinopathy (DR) severity levels from retinal fundus images using convolutional neural networks and transfer learning. The goal is to build a reliable multi-class classifier that can support early screening and grading of diabetic retinopathy.

🎯 Problem Statement

Diabetic Retinopathy is a diabetes-related eye disease that damages retinal blood vessels and can lead to blindness if not detected early. Manual grading of retinal images is time-consuming and requires trained ophthalmologists.

This project aims to:

1)classify retinal fundus images into 5 severity levels

2)evaluate performance on imbalanced clinical data

3)analyze model behavior across minority disease classes

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

1)Strong class imbalance, especially in Severe and Proliferative classes

2)Large variations in brightness, contrast, and image quality

🔍 Exploratory Data Analysis

Before training, I analyzed:

✔ Class Distribution

The dataset is highly imbalanced, with far fewer samples in advanced disease categories.

✔ Visual Inspection

Random samples from each class were visualized to inspect:

1)lesion patterns

2)illumination differences

3)background artifacts

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

1)Image resizing to 224 × 224

2)Model-specific normalization:

3)ResNet preprocessing for ResNet model

4)EfficientNet preprocessing for EfficientNet model

5)Horizontal flipping for augmentation

6)To improve I/O speed, images were copied from Google Drive to local Colab storage before training.

🧠 Models and Training
🔹 Baseline Model — ResNet50

1)Pretrained on ImageNet

2)Backbone frozen

3)Custom classifier head:

4)Global Average Pooling

5)Dense (256)

6)Dropout (0.5)

7)Softmax (5 classes)

Purpose: evaluate how generic visual features perform on retinal images.

🔹 Improved Model — EfficientNetB0

EfficientNet was chosen due to:

better parameter efficiency

stronger feature extraction

Stage 1: Feature Extraction

1)Backbone frozen

2)Only classifier trained

Stage 2: Fine-Tuning

1)Upper convolution layers unfrozen

2)Lower layers kept frozen

3)Reduced learning rate

This allows adaptation to retinal lesion patterns while preserving general visual features.

📈 Evaluation Results
🔹 Baseline (ResNet50)

1)Accuracy: ~78%

2)Very low recall for:

3)Severe DR

4)Proliferative DR

This indicates strong bias toward majority classes due to imbalance.

🔹 Improved Model (EfficientNetB0 + Fine-Tuning)

1)Accuracy: ~83%

2)Improved recall for minority classes

3)Better macro-averaged F1 score

Remaining confusion mainly occurs between:

1)Moderate vs Severe DR

2)Severe vs Proliferative DR

These categories also show overlap in clinical grading.

🧠 Analysis and Limitations

1)Severe class imbalance still limits recall for advanced disease stages

2)Borderline clinical cases are difficult even for human experts

3)Model trained on single dataset — generalization to other hospitals not guaranteed

These limitations motivate further research into:

1)class-balanced loss functions

2)multi-dataset training

3)lesion-aware attention mechanisms

💾 Model Saving

1)Trained models are saved for:

2)inference experiments

3)Grad-CAM visualization

further fine-tuning

Both baseline and final models are stored for future comparison.

🚀 Future Work

Planned extensions:

✔ Explainability

1)Grad-CAM to verify that predictions focus on lesion regions instead of background artifacts

✔ Class Imbalance Handling

Focal loss

2)Class-weighted loss functions

✔ Generalization

3)Evaluation on external datasets (IDRiD, Messidor)

✔ Clinical Relevance

4)Binary DR screening → severity grading pipeline

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
