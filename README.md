MNIST Handwritten Digit Recognition (CNN)
This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify handwritten digits from the world-famous MNIST dataset. The model achieves high accuracy and has been tested with real-world custom handwritten inputs.

🚀 Key Achievements
Test Accuracy: Successfully achieved 99.13% accuracy on the test dataset.

Overfitting Prevention: Effectively managed the gap between training and validation performance using Dropout and Early Stopping.

Real-world Generalization: The model accurately predicts digits from custom, noisy, hand-drawn images.

🏗️ Model Architecture
The network is built using TensorFlow/Keras with the following structure:

Convolutional Layers: To extract spatial features from the 28x28 grayscale images.

Max Pooling: For spatial downsampling.

Flatten & Dense Layers: To interpret features and classify them into 10 categories (0-9).

Dropout: To ensure the model generalizes well to unseen data.

📊 Performance Analysis
1. Training Progress (Accuracy & Loss Curves)
As shown in the learning curves below, the model's training accuracy improved consistently while the validation accuracy remained high. The loss curve indicates a healthy convergence.

2. Confusion Matrix
The confusion matrix highlights the model's precision across all 10 digits. The high diagonal values indicate nearly perfect classification, with only minor confusion between visually similar digits (e.g., 4 and 9).

3. Real-world Testing (Handwritten Prediction)
The model was tested using a custom hand-drawn digit. Despite the noise and variations in the input image, the model correctly predicted the number.

Input Image: Hand-drawn digit '1'

Prediction Result: 1

🛠️ Tech Stack & Tools
Programming Language: Python

Deep Learning Framework: TensorFlow / Keras

Visualization: Matplotlib, Seaborn

Image Processing: OpenCV

Environment: Kaggle Notebooks

📂 How to Run
Clone this repository.

Install dependencies: pip install tensorflow matplotlib seaborn opencv-python.

Open the .ipynb notebook in Kaggle or Jupyter and run all cells.

Upload your own 28x28 grayscale images to see the model in action!

Developed by: Sunny
