# Streamlit Machine Learning Classifier

## Project Overview

This project was created for my AI Architecture and Design (ARTI 405) course at Ferris State University. The goal was to build and improve a simple machine learning web application using Python and Streamlit.

The application allows users to select different datasets, choose a machine learning classifier, and adjust hyperparameters to see how they affect model performance.

## Features

The application includes three datasets:
- Iris
- Breast Cancer
- Wine

Users can choose between three machine learning classifiers:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

Each classifier has adjustable hyperparameters:

**KNN:** Number of Neighbors (K) and Leaf Size

**SVM:** C, Degree, and Kernel

**Random Forest:** Max Depth, Number of Estimators, and Min Samples Split

The application trains the selected model and displays its accuracy, classification report, confusion matrix, and a PCA scatter plot to help visualize the dataset.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Matplotlib
- GitHub
- Visual Studio Code

## Running the Application

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run Interface.py
```

The application will open in a web browser where users can interact with the different classifiers and their settings.

## What I Learned

Through this project, I gained more experience using Streamlit to create interactive machine learning applications. I also learned how different hyperparameters can affect model behavior and performance, how to visualize classification results, and how to use GitHub to manage and share my code.

## Course Information

**Course:** ARTI 405 – AI Architecture and Design

**University:** Ferris State University
