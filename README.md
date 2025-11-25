🐾 Wildlife Animal Classifier (ResNet-18)

An AI-powered wildlife animal classification system trained on a YOLO-formatted dataset containing 54 animal species. This project uses OpenCV for automated image cropping and PyTorch for training a fine-tuned ResNet-18 Convolutional Neural Network, achieving 91.07% accuracy.

FEATURES
- Automated dataset preprocessing
- Deep Learning Model (ResNet-18)
- High Performance (94.67% train acc, 91.07% test acc)
- Comprehensive Evaluation

MODEL ARCHITECTURE
- ResNet-18 (Transfer Learning)
- Frozen convolution layers
- Custom FC layer for 54 classes
- Adam optimizer, CE Loss

DATASET
- Kaggle Wildlife Dataset
- YOLO format
- Image cropping via OpenCV

TECHNOLOGIES
Python, PyTorch, Torchvision, OpenCV, NumPy, Matplotlib, Seaborn

PROJECT STRUCTURE
Resnet_Wildlife_Classifier/
│── preprocessing/
│── train/
│── test/
│── resnet18_wildlife.pth
│── model_training.ipynb
│── evaluation.ipynb
│── README.txt

TRAINING CODE SNIPPET
model = models.resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 54)

EVALUATION
- 91.07% accuracy
- Confusion matrix
- Class-wise accuracy
- Prediction samples

DataSet:-
https://www.kaggle.com/datasets/banuprasadb/wildlife-dataset :- Kaggle
