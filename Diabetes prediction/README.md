# Diabetes Prediction using Neural Network

## Overview
This project implements a neural network model to predict whether a patient has diabetes or not based on medical attributes. The model is built using Keras/TensorFlow and trained on the Pima Indians Diabetes Dataset.

## Project Description
The model predicts diabetes diagnosis (binary classification: 0 = no diabetes, 1 = diabetes) using 8 medical features:

1. Number of times pregnant
2. Plasma glucose concentration at 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (BMI: weight in kg/(height in m)²)
7. Diabetes pedigree function
8. Age (years)

## What's Included

- **train.py** - Script to train the neural network model from scratch
- **test.py** - Script to load the trained model and make predictions
- **model.json** - Saved model architecture in JSON format
- **model.h5** - Saved model weights in HDF5 format
- **pima-indians-diabetes.csv** - Training dataset containing 768 samples

## Environment & Dependencies

### Required Python Version
- Python 3.6 or higher

### Required Libraries
Install the following packages using pip:

```bash
pip install numpy
pip install keras
pip install tensorflow
```

Or install from a requirements file (if available):
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Training the Model
To train a new model from scratch:

```bash
python train.py
```

This will:
- Load the diabetes dataset
- Build a neural network with 3 layers
- Train the model for 40 epochs with batch size of 10
- Evaluate accuracy on the training data
- Save the model architecture to `model.json` and weights to `model.h5`

Expected output: Model accuracy (typically around 75-77%)

### 2. Testing the Model
To load the pre-trained model and make predictions:

```bash
python test.py
```

This will:
- Load the saved model from `model.json` and `model.h5`
- Make predictions on sample data
- Display 5 predictions (samples 5-9) with their expected results

## Model Architecture

The neural network consists of:
- **Input Layer**: 8 features
- **Hidden Layer 1**: 12 neurons, ReLU activation
- **Hidden Layer 2**: 8 neurons, ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation (binary classification)

**Compilation**:
- Loss function: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

## Getting Started for New Users

1. **Install Python** (if not already installed): Download from [python.org](https://www.python.org/)

2. **Install dependencies**:
   ```bash
   pip install numpy keras tensorflow
   ```

3. **Clone or download this project** to your local machine

4. **Navigate to the project directory**:
   ```bash
   cd "Diabetes prediction"
   ```

5. **Run the test script** to see predictions (uses pre-trained model):
   ```bash
   python test.py
   ```

6. **Or train a new model**:
   ```bash
   python train.py
   ```

## Notes

- The pre-trained model has already been trained and saved, so you can run `test.py` immediately
- Training a new model will overwrite the existing `model.json` and `model.h5` files
- The dataset contains medical data from the Pima Indian community
- This is a binary classification problem (diabetes or no diabetes)

## Dataset Source
Pima Indians Diabetes Dataset - A classic dataset in machine learning research

## License & Usage
This project is for educational purposes. Feel free to modify and extend it for your learning.
