# ML-Projects

This repository contains a curated collection of end-to-end Machine Learning and Deep Learning projects developed as part of my structured journey in Artificial Intelligence. Each project focuses on practical implementation, clean architecture, reproducible workflows, and production-oriented thinking.

The goal is not just to train models, but to design complete systems — from data preprocessing and feature engineering to evaluation, optimization, and deployment.


## Repository Overview

This repository currently includes the following projects:


### 1. Energy Consumption Forecasting & Anomaly Detection

An end-to-end time-series forecasting system designed to predict hourly energy consumption and detect anomalous usage patterns.

**Key Highlights**

* Time-based train/validation/test split to prevent data leakage
* Lag and rolling-window feature engineering
* Model comparison (Ridge Regression, Random Forest, Gradient Boosting)
* Residual-based anomaly detection using percentile thresholding
* Structured evaluation using MAE, RMSE, and MAPE
* Modular pipeline packaging with CLI scoring support

**Core Technologies**

* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib
* Joblib


### 2. Fraud Detection System

A production-oriented imbalanced classification system for detecting fraudulent credit card transactions (284,000+ records, <1% fraud rate).

**Key Highlights**

* Feature engineering and data preprocessing
* Class imbalance handling (class weights and evaluation trade-offs)
* Model comparison (Logistic Regression, Random Forest, Gradient Boosting)
* Hyperparameter tuning with cross-validation
* Precision–recall optimization and threshold tuning
* Packaged inference pipeline with saved model artifacts

**Core Technologies**

* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Joblib


### 3. CIFAR-10 CNN Classifier

A deep learning image classification project implemented using Convolutional Neural Networks in PyTorch.

**Key Highlights**

* Custom CNN architecture implementation
* Training and evaluation workflow
* Hyperparameter experimentation
* Model saving and loading
* Modular project structure

**Core Technologies**

* Python
* PyTorch
* NumPy
* Matplotlib


### 4. CIFAR-10 Streamlit Application

A lightweight deployment application serving the trained CNN model for interactive image classification.

**Key Highlights**

* Model loading and inference pipeline
* Streamlit-based user interface
* Real-time image prediction

**Core Technologies**

* Python
* Streamlit
* PyTorch


### 5. Heart Disease Prediction Project

A supervised learning system for predicting heart disease risk using structured clinical data.

**Key Highlights**

* Exploratory Data Analysis
* Feature preprocessing
* Model training and comparison
* Performance evaluation

**Core Technologies**

* Python
* Scikit-learn
* Pandas
* Matplotlib


### 6. House Price Prediction Project

A regression-based machine learning project for predicting housing prices using structured tabular data.

**Key Highlights**

* Data cleaning and preprocessing
* Feature engineering
* Regression modeling
* Model evaluation using standard error metrics

**Core Technologies**

* Python
* Scikit-learn
* Pandas
* NumPy


## Technical Stack

Across these projects, the primary tools and frameworks used include:

* Python
* NumPy
* Pandas
* Scikit-learn
* PyTorch
* Matplotlib
* Streamlit
* Jupyter Notebook


## Project Structure

Each project follows a clean and modular structure, typically including:

* `data/` for datasets (large files excluded via .gitignore)
* `notebooks/` for experimentation and analysis
* `src/` for modular Python scripts
* `models/` for trained model artifacts (ignored if large)
* `requirements.txt` for dependency management
* Project-specific `README.md` for documentation

This structure emphasizes reproducibility, clarity, and maintainability.


## Objective

The objective of this repository is to:

* Strengthen applied Machine Learning and Deep Learning skills
* Build industry-relevant, portfolio-ready systems
* Practice structured experimentation and performance evaluation
* Develop deployable ML pipelines
* Prepare for advanced research and professional AI/ML roles


## Author

Saiful Islam Shihab

Bachelor of Computer Science (Artificial Intelligence Major)

La Trobe University

Melbourne, Australia
