# Generated from: feature_extraction.ipynb
# Converted at: 2026-01-13T12:27:44.477Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import cv2
import numpy as np
import os
import pandas as pd
import math
import csv
import random

def is_leaf_image(img,min_area=2000,min_solidity=0.35,min_green=40):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours) == 0:
        return False

    contour = max(contours,key=cv2.contourArea)
    area = cv2.contourArea(contour)

    if area < min_area:
        return False

    hull=cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return False


    solidity = area/hull_area
    if solidity < min_solidity:
        return False

    mean_green = np.mean(img[:,:,1])
    if mean_green < min_green:
        return False


    return True

    

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None


    if not is_leaf_image(img):
        return None

    # SHAPE
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None


    contour = max(contours,key=cv2.contourArea)
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area if hull_area > 0 else 0
    damaged_area = hull_area - area


    #-COLOR

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    mean_hue = np.mean(H)
    mean_sat = np.mean(S)
    mean_val = np.mean(V)

    mean_green = np.mean(img[:, :, 1])
    


    return [area,perimeter,solidity,damaged_area,mean_hue,mean_sat,mean_val,mean_green]

    # CSV DATASET 

dataset_path = "/home/ajinkya/RK Mam/Dataset"
csv_file = "/home/ajinkya/RK Mam/leaf_features.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Area", "Perimeter", "Solidity", "DamagedArea",
        "MeanHue", "MeanSaturation", "MeanValue",
        "MeanGreen", "Label"
    ])

    for label in ['Healthy', 'Unhealthy']:
        folder = os.path.join(dataset_path, label)

        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(folder, file)
                features = extract_features(image_path)

                if features is not None:
                    writer.writerow(features + [label])

print("CSV dataset created: leaf_features.csv")





df = pd.read_csv("leaf_features.csv")
df.head()

print(df["Label"].unique())

# Encode the Labels

y = []
for label in df["Label"]:
    if label == "Healthy":
        y.append(0)
    elif label == "Unhealthy":
        y.append(1)

    else:
        raise ValueError("Invalid label Found")

print("First Label:",y[0])

#seprate feature 

x = []
features_cols = df.columns[:-1]

for _,row in df.iterrows():
    features = []
    for col in features_cols:
        features.append(float(row[col]))
    x.append(features)

print("Samples:",len(x))
print("Feature per sample:",len(x[0]))

mean = [sum(col)/len(col) for col in zip(*x)]
stds = [(sum((v-m) ** 2 for v in col)/len(col)) ** 0.5
        for col,m in zip(zip(*x),mean)]

x = [[(xi - m) / s for xi ,m,s, in zip (row , mean,stds)]
     for row in x ]
print(mean)
print(stds)

# Shuffle data

data = list(zip(x,y))
random.shuffle(data)
x,y = zip(*data)

x = list(x)
y = list(y)

print("features:" ,x[:5])
print("Labels: ",y[:5])


# Train - Test Split

train_ratio = 0.8
total_sample = len(x)
train_size = int(train_ratio * total_sample)

x_train = x[:train_size]
y_train = y[:train_size]

x_test = x[train_size:]
y_test = y[train_size:]

print("Train samples: ",len(x_train))
print("Test samples : ",len(x_test))

# initialize theta randomly

n_features = len(x_train[0])
theta = [random.uniform(-0.01,0.01) for _ in range(n_features + 1)]

print("Initial theta values : ",theta)

def sigmoid(z):
    if z >= 0:
        return 1/(1+math.exp(-z))
    else:
        return math.exp(z)/(1 + math.exp(z))

#prediction

def predict_prob(x,theta):
    x_with_bias = [1] + x
    z = 0
    for i in range(len(theta)):
        z += theta[i] * x_with_bias[i]

    return sigmoid(z)


def loss(y,p):
    epsilon = 1e-15
    p=max(min(p,1 - epsilon),epsilon)
    return -(y*math.log(p) + (1-y) * math.log(1-p))

    

# training loop

learning_rate =0.002
epochs = 500

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(x_train)):
        x = x_train[i]
        y_true = y_train[i]

        x_with_bias = [1] + x
        p = predict_prob(x,theta)
        error = p - y_true

        for j in range(len(theta)):
            theta[j] -= learning_rate * error * x_with_bias[j]

        total_loss += loss(y_true,p)


    avg_loss = total_loss/len(x_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

# test the model 
correct = 0

for i in range(len(x_test)):
    p = predict_prob(x_test[i],theta)
    y_pred = 1 if p >= 0.5 else 0

    if y_pred == y_test[i]:
        correct +=1


accuracy = correct/len(x_test)
print("Accuracy :",accuracy)

#confussion Matrix

tp = 0
tn = 0
fp = 0
fn = 0

for i in range(len(x_test)):
    p = predict_prob(x_test[i],theta)
    y_pred = 1 if p >= 0.5 else 0
    y_true = y_test[i]

    if y_true == 1 and y_pred == 1:
        tp +=1
    elif y_true == 0 and y_pred == 0:
        tn +=1
    elif y_true == 0 and y_pred == 1:
        fp +=1
    elif y_true == 1 and y_pred == 0:
        fn +=1


print("Confusion Matrix :")
print([[tn,fp],
        [fn,tp]])

accuracy =  (tp+tn)/(tp + tn + fp + fn)
print("Accuracy:", accuracy)


import pickle

model_data = {
    "theta": theta,
    "mean": mean,
    "stds": stds
}

with open("LR_plant_model.pkl", "wb") as f:
    pickle.dump(model_data, f)