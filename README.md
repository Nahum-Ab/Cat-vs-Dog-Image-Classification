# Cat vs Dog Image Classification using SVM

This project implements a Support Vector Machine (SVM) model to classify images of cats and dogs. The model is trained on a dataset of images and can predict whether a given image contains a cat or a dog.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

The goal of this project is to develop a machine learning model that can accurately classify images of cats and dogs. The model uses a Support Vector Machine (SVM) algorithm, which is well-suited for binary classification tasks.

## Dataset

The dataset used for this project is the **Dogs vs. Cats** dataset from Kaggle. You can download the dataset from [here](https://www.kaggle.com/c/dogs-vs-cats/data).
Make sure to replace the path in the code with the actual path to your dataset.

## Installation

To run this project, you'll need to have Python installed along with the following libraries:

- OpenCV
- NumPy
- scikit-learn

You can install the required libraries using pip:

```bash
pip install opencv-python numpy scikit-learn
```

## Usage

To use this project, follow these steps:

1. **Clone the Repository**: Start by cloning the repository to your local machine. Open your terminal and run:

   ```bash
   git clone https://github.com/Nahum-Ab/cat-dog-classification.git
   cd cat-dog-classification
   ```

2. **Set Up the Dataset**: Download the Dogs vs. Cats dataset from Kaggle. Extract the contents and ensure that the images are organized in the following structure:

   ```text
   /path/to/dataset/train/
    ├── cat.1.jpg
    ├── cat.2.jpg
    ├── dog.1.jpg
    ├── dog.2.jpg
    └── ...
   ```

3. **Update the Dataset Path**: Open the classify.py file and update the data_dir variable to point to the location of your dataset:

   ```python
   data_dir = 'C:\\Users\\acer\\OneDrive\\Desktop\\Classify_images\\train'  # Update this path to where the dataset is located, Please becarefull here cause the code works if you specifiy the correct PATH!
   ```

4. **Install Required Libraries**: Make sure you have the necessary Python libraries installed. You can install them using pip:

   ```bash
   pip install opencv-python numpy scikit-learn
   ```

5. **Run the code, See the result**: Execute the script to train the model and evaluate its performance:

   ```bash
   python classify.py
   ```



