WiSe2425-TNN-Assignment-C
# WiSe2425-TNN-Assignment-B

This project implements and trains various **Multi-Layer Perceptron (MLP)** models to perform **encoder-decoder tasks** with different configurations, including 8-2-8, 8-3-8, 50-2-50, and 64-2-64 architectures. The goal of each model is to compress input data through a bottleneck layer and reconstruct the input at the output layer.

## Table of Contents

- [Project Structure](#project-structure)
- [Data Files](#data-files)
- [Requirements](#requirements)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
  - [src/model.py](#srcmodelpy)
  - [src/train.py](#srctrainpy)
  - [src/evaluate.py](#srcevaluatepy)
  - [src/utils.py](#srcutilspy)
- [Training and Evaluation](#training-and-evaluation)
- [Learning Curves](#learning-curves)

## Project Structure
```
WiSe2425-TNN-Assignment-C/
 ├── data/
 │   ├── PA-A_training_data_01.txt # Data
 │   ├── PA-A_training_data_02.txt # Data
 │   ├── PA-A_training_data_03.txt # Data
 │   ├── PA-A_training_data_04.txt # Data
 │   ├── PA-A_training_data_05.txt # Data
 │   └── PA-A_training_data_06.txt # Data
 ├── logs/
 │ ├── learning_curve.txt 
 ├── reports/
 │ ├── performance_report.txt
 ├── src/
 │ ├── evaluate.py
 │ ├── main.py
 │ ├── model.py
 │ ├── train.py
 │ └── utils.py
 ├── .gitignore
 ├── README.md
 └── requirements.txt
```
## Installation

To run this project locally, you'll need to have **Python 3.9+** installed.

1. Clone this repository:
```
git clone git@github.com:s93nnguy/WiSe2425-TNN-Assignment-C.git
cd WiSe2425-TNN-Assignment-C
```
2. Install the required Python packages using `pip`:
```
pip install -r requirements.txt
```
### Running the Training and testing the model
To train the model, run the following command from the root directory:
```
python src/main.py
```
This will:
- Loading the data
  - Loading data from one file in folder `data/`
  - Read p, n, m from header of the file
  - Separate data into 2 parts with rate (default rate = 0.8): train data and test data
- Training
  - if p = k, init model and calculate weight 
  - Train the RBF model with gradient to update weight and K-Mean to calculate center and width
  - Print the loss at each iteration.
  - Save the learning curve to `logs/`
  - Save running time (train process)
- Testing
  - Evaluate the test data
  - Save the Mean Squared Error (MSE) on the test dataset
- Moore-Penrose-Pseudo-Inverse
  - Using Moore-Penrose-Pseudo-Inverse to calculate weight
  - Save running time and mse when using Moore-Penrose-Pseudo-Inverse.
- Compare performance (running time and loss) and save to file `logs/learning_curve.txt`
