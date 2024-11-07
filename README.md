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
 │ ├── 
 ├── logs/
 │ ├── 
 ├── reports/
 │ ├── 
 ├── scripts/
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
  - Generate training dataset and store in folder `data/` 
  - Separate it into 2 parts with rate (default rate = 0.8): train data and test data
- Training
  - Train the model using the backpropagation algorithm.
  - Print the loss at each epoch.
  - Print the resulting input and output values to console.
  - Save the learning curve to `logs/` and plot the learning curve and save visualized learning curve to folder `reports/`
- Testing
  - Evaluate the test data
  - Print the Mean Squared Error (MSE) on the test dataset.
- Print output
  - Print the resulting input and output values after evaluating to console and save to file `reports/output_<task_name>.txt`
  - Visualize hidden states to figures saved to `reports/hidden_states_<task_name>.png`
