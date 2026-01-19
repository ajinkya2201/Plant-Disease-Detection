import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------
# Image validation
# --------------------------------------------------
def is_valid_image(file):
    try:
        img = Image.open(file)
        img.verify()
        return True
    except Exception:
        return False


# --------------------------------------------------
# Logistic regression helpers
# --------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_predict(x, theta):
    """
    x     : (n_features,)
    theta : (n_features + 1,)
    """
    x = np.insert(x, 0, 1)   # add bias term
    z = np.dot(theta, x)
    return sigmoid(z)


# --------------------------------------------------
# Leaf feature extraction (8 features)
# --------------------------------------------------
def extract_leaf_features(image_path):
    """
    Returns numpy array of 8 features:
    [area, perimeter, solidity, damaged_area,
     mean_hue, mean_saturation, mean_value, mean_green]
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to read image")

    img = cv2.resize(img, (256, 256))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No leaf detected")

    cnt = max(contours, key=cv2.contourArea)

    # 1. Area
    area = cv2.contourArea(cnt)

    # 2. Perimeter
    perimeter = cv2.arcLength(cnt, True)

    # 3. Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0

    # 4. Damaged area
    damaged_area = hull_area - area

    # HSV features
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = thresh == 255

    # 5–7 Mean HSV
    mean_hue = h[mask].mean()
    mean_saturation = s[mask].mean()
    mean_value = v[mask].mean()

    # 8. Mean green channel
    mean_green = img[:, :, 1][mask].mean()

    return np.array([
        area,
        perimeter,
        solidity,
        damaged_area,
        mean_hue,
        mean_saturation,
        mean_value,
        mean_green
    ])
