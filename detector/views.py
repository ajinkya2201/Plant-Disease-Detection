from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np

from .model_loader import load_model
from .utils import (
    is_valid_image,
    extract_leaf_features,
    logistic_predict
)

# --------------------------------------------------
# Load trained model parameters ONCE
# --------------------------------------------------
model_data = load_model()

theta = np.array(model_data['theta'])   # shape: (9,)
mean = np.array(model_data['mean'])     # shape: (8,)
stds = np.array(model_data['stds'])     # shape: (8,)

# Debug (can be removed later)
print("Theta shape:", theta.shape)
print("Mean shape:", mean.shape)
print("Stds shape:", stds.shape)


# --------------------------------------------------
# Home View
# --------------------------------------------------
def home(request):

    if request.method == 'POST' and request.FILES.get('leaf_image'):

        image = request.FILES['leaf_image']

        # 1. Validate image
        if not is_valid_image(image):
            return render(request, 'detector/home.html', {
                'error': 'Invalid image file. Please upload a valid leaf image.'
            })

        # 2. Save uploaded image
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        image_path = fs.path(filename)
        file_url = fs.url(filename)

        try:
            # 3. Extract REAL leaf features (8 features)
            x = extract_leaf_features(image_path)   # shape: (8,)

            # 4. Normalize features (same as training)
            x = (x - mean) / stds

            # 5. Logistic regression prediction
            prob = logistic_predict(x, theta)

            # 6. Class decision
            result = "Diseased Leaf" if prob >= 0.3 else "Healthy Leaf"
            if result == "Healthy Leaf":
                 description = (
        "The leaf appears healthy with no visible signs of disease. "
        "Maintain proper irrigation, sunlight, and regular monitoring "
        "to ensure continued plant health."
    )
            else:
                 description = (
        "The leaf shows symptoms of disease. This may affect plant growth "
        "and yield. It is recommended to isolate the plant, monitor nearby plants, "
        "and apply appropriate treatment or consult an agricultural expert."
    )


            return render(request, 'detector/home.html', {
    'file_url': file_url,
    'result': result,
    'prob': round(float(prob), 3),
    'description': description
            })

        except Exception as e:
            # Any error in feature extraction or prediction
            return render(request, 'detector/home.html', {
    'file_url': file_url,
    'result': result,
    'prob': round(float(prob), 3),
    'description': description
})


    # GET request
    return render(request, 'detector/home.html')
