

import cv2
import numpy as np
import os
import pandas as pd
import math
import csv

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


y = []
for label in df["Label"]:
    if label == "Healthy":
        y.append(0)
    elif label == "Unhealthy":
        y.append(1)

    else:
        raise ValueError("Invalid label Found")

print("First Label:",y[:20])


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

data = list(zip(x,y))
random.shuffle(data)
x,y = zip(*data)

x = list(x)
y = list(y)

print("features:" ,x[:5])
print("Labels: ",y[:5])


# Split

train_ratio = 0.8
total_sample = len(x)
train_size = int(train_ratio * total_sample)

x_train = x[:train_size]
y_train = y[:train_size]

x_test = x[train_size:]
y_test = y[train_size:]

print("Train samples: ",len(x_train))
print("Test samples : ",len(x_test))

# add bias to features

x_train_bias = [[1] + row for row in x_train]
x_test_bias = [[1] + row for row in x_test]

print("feature with bias :",x_train_bias[0])
print("Total features now : " , len(x_train_bias[0]))


# initialize theta with random number

num_features = len(x_train_bias[0])

theta = [random.uniform(-0.01,0.01) for _ in range(num_features)]

print("Number of weights :",len(theta))
print("initial values of theta :",theta)

# sigmoid 
#convert into probability range [0,1]

def sigmoid(z):
    return 1/ (1 + math.exp(-z))

# print(sigmoid(0))
# print(sigmoid(2))
# print(sigmoid(-2))


# connect sigmoid to theta.T * x
#this compute y^ = sigmoid(theta.T*x)

def predict_prob_single(x,theta):

    assert isinstance(x,list)
    assert isinstance(x[0],(int,float))
    
    z = 0.0
    for i in range(len(theta)):
        z += theta[i] * float(x[i])
    return sigmoid(z)

p = predict_prob_single(x_train_bias[0] ,theta)
print("predicted probability :",p)

#loss for one smaple

def loss_single(y,y_pred):
    eps = 1e-9
    return -(y * math.log(y_pred + eps)+(1-y) * math.log(1-y_pred+eps))

y_true = y_train[0]  #actual label
p = predict_prob_single(x_train_bias[0],theta)

l=loss_single(y_true,p)

print("True label : ",y_true)
print("predicted probability: ",p)
print("Loss : ",l)

# gradient descent for 1 sample 
def gradient_single(x,y,y_pred):
    grads =[]
    for i in range(len(x)):
        grad_i = (y_pred - y) * x[i]
        grads.append(grad_i)
    return grads


x_sample = x_train_bias[0]
y_true = y_train[0]
y_pred = predict_prob_single(x_sample, theta)

grads = gradient_single(x_sample, y_true, y_pred)

print("Gradients for one sample:")
print(grads)
print("Number of gradients:", len(grads))


# gradient descent update rule 
# update theta

def update_theta(theta, grads,lr):
    new_theta=[]
    for i in range(len(theta)):
        update_value = theta[i] - lr * grads[i]
        new_theta.append(update_value)

    return new_theta


#apply update on one sample

lr = 0.1

x_sample = x_train_bias[0]
y_true = y_train[0]
y_pred = predict_prob_single(x_sample, theta)

grads = gradient_single(x_sample, y_true, y_pred)

theta_new = update_theta(theta, grads, lr)

print("Old theta:", theta)
print("\n")
print("New theta:", theta_new)


#repeate the update for all training sample

def train_one_epoch(x_train,y_train,theta,lr):
    for i in range(len(x_train)):
        x_sample = x_train[i]
        y_true = y_train[i]

        y_pred = predict_prob_single(x_sample,theta)
        grads = gradient_single(x_sample,y_true,y_pred)
        theta = update_theta(theta,grads,lr)

    return theta


lr = 0.01
theta = train_one_epoch(x_train_bias,y_train,theta,lr)

print("theta after one epoach:",theta)



# train on multiple epoch


def train_model(x_train,y_train,theta,lr,epochs):
    loss_history = []
    
    for epoch in range(epochs):
        theta = train_one_epoch(x_train,y_train,theta,lr)

        #compute loss after one epoch
        total_loss = 0.0
        for i in range(len(x_train)):
            y_pred = predict_prob_single(x_train[i],theta)
            total_loss += loss_single(y_train[i],y_pred)

        avg_loss = total_loss/len(x_train)
        loss_history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss = {avg_loss}")

    return theta,loss_history

    
lr = 0.01
epochs = 50
theta,loss_history = train_model(x_train_bias,y_train,theta,lr,epochs)
print(" theta: ",theta)
        

import matplotlib.pyplot as plt
epochs = range(len(loss_history))

plt.figure()
plt.plot(epochs, loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.show()


#predict the probability on test data 

def predict_all(X, theta):
    probs = []
    for x in X:
        probs.append(predict_prob_single(x, theta))
    return probs


#convert the probability to class label
def predict_labels(probs, threshold=0.5):
    labels = []
    for p in probs:
        labels.append(1 if p >= threshold else 0)
    return labels


#compare predictions with actual value
y_test_prob = predict_all(x_test_bias, theta)
y_test_pred = predict_labels(y_test_prob)

print("First 10 predictions:", y_test_pred[:15])
print("First 10 actual labels:", y_test[:15])


def confusion_matrix(y_true, y_pred):
    tp = tn = fp = fn = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1

    return tp, tn, fp, fn

tp, tn, fp, fn = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix:")
print("[[TN FP]")
print(" [FN TP]]")

print([[tn, fp],
       [fn, tp]])



accuracy = (tp + tn) / (tp + tn + fp + fn)
accuracy
