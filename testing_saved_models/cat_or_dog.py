import cv2
import numpy as xp
import matplotlib.pyplot as plt
from models.feedforward_nn import Model

model = Model.load("../saved_models/model_cat_or_dog.pickle")

def load_single_image(path, img_size=128):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))

    x = xp.array(img_resized, dtype=xp.float32) / 255.0
    x = x.transpose(2, 0, 1)
    x = x[xp.newaxis, ...]

    return img, x

image_paths = [
    "../data/PetImages/Cat/3698.jpg",
    "../data/PetImages/Cat/8355.jpg",
    "../data/PetImages/Dog/8040.jpg",
    "../data/PetImages/Dog/9259.jpg",
]

results = []

for path in image_paths:
    img_rgb, x = load_single_image(path)
    y_pred = model(x)
    prob = float(y_pred[0, 0])

    label = "DOG" if prob >= 0.5 else "CAT"
    results.append((img_rgb, label, prob))

plt.figure(figsize=(12, 6))

for i, (img, label, prob) in enumerate(results):
    plt.subplot(1, 4, i + 1)
    plt.imshow(img)
    if label == "DOG":
        plt.title(f"{label}\n{prob:.2f}")
    else:
        plt.title(f"{label}\n{(1-prob):.2f}")
    plt.axis("off")

plt.tight_layout()
plt.show()

"""
Results are saved at saved_models/cat_dog_prediction_results.jpg
"""