
# Plant Diesase Detection

A Machine Learning based **Plant Diesase Detection** that classifies the leaf image as **Healthy** and **Unhealthy** using Image processing , Logistic Regression(without sklearn) and Django web interface.


## Project Overview

This project demonstrates the end-to-end workflow of a machine learning application:


* Leaf Image Processing
* Feature Extraction
* Logistic Regression implementation without using sklearn
* Model Training and Evaluation
* Web- Based Prediction using Django


## Machine Learning approach

+ Algorithm : Logistic Regression
+ Loss Function : Binary Cross-Entropy
+ Optimization : Stochastic Gradient Descent(SGD)
+ Gradient Used : 

            dL/dw = ( p-y ) * x
            dL/db = ( p-y )


## Dataset 

#### Leaf Images are divided into :

  + Healthy
  + Unhealthy


#### Image are converted into numerical features and stored in
  + leaf_features.csv



## Workflow


+ ### Image Preprocessing

  * Image loading

  * Resizing

  * Color space conversion(RGB -> HSV)


+ ### Feature Extraction

  * Shape features (area, perimeter, etc.)

  * Color features (mean hue, saturation, value, green channel)

+ ### Dataset Creation

  * Features stored in CSV format
  * each row represnt one leaf image
  * Last column represents Label(Healthy/Unhealthy)

+ ### Label Encoding
  * Label are manually encoded:
    * Healthy -> 0
    * Unhealthy -> 1
  

+ ### Feature Scaling

  * Standardization:

           (x − mean) / std


+ ### Data shuffling and Train-Test Split
  * Data is shuffle to avoid bias
  * Data is split into:
    * 80% Training
    * 20% Testing

+ ### Logistic Regression (without sklearn)
  ##### Model Equation:

                  z = b + W1X1 + W2X2 + ...... + WnXn
                
                  sigmoid(z) = 1 / (1 + e^(-z))

                  p = sigmoid(z)

   p is probability of the leaf being Unhealthy

   sigmoid map value between 0 and 1

+ ### Loss Function

          L = -[y * log (p) + (1 - y) * log(1 - p)]



+ ### Gradient Descent
  * Model Parameters are update using Stochastic Gradient Descent(SGD)

    Derived Gradient:

                  dL/dw = ( p-y ) * x
                  dL/db = ( p-y )
 

+ ### Model Training

  * Training is performed for multiple epoch
  * Loss is Monitored


+ ### Prediction

    For unseen leaf data:

    + Probability is computed using sigmoid

    + A threshold (0.5) is used:

        * ≥ 0.5 → Unhealthy

        * < 0.5 → Healthy
+ ### Model Evaluation

  Model performance is evaluated using:

    + Accuracy

    + Confusion Matrix (TP, TN, FP, FN)

## Technologies Used:
    * Python
    * OpenCV(image Processing)
    * Numpy
    * Pandas
    * Math and Random Module
    * Django

## How to Run

+ Train the Model

  ~ python plant_model.py

+ Run Django server

  ~python manage.py runserver

+ Open In browser

  http://127.0.0.1:8000/

