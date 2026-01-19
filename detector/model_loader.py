# import pickle
# import os
# from django.conf import settings

# MODEL_PATH = os.path.join(settings.BASE_DIR, 'detector', 'LR_plant_model.pkl')

# def load_model():
#     with open(MODEL_PATH, 'rb') as f:
#         data = pickle.load(f)

#     # If saved as dictionary
#     if isinstance(data, dict):
#         return data
#     else:
#         return {"model": data}

import pickle
import os
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, 'detector', 'LR_plant_model.pkl')

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)

    print("TYPE:", type(data))

    if isinstance(data, dict):
        print("DICT KEYS:", data.keys())

    return data

