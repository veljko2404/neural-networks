import os
import cv2
from backend.backend import xp
import matplotlib.pyplot as plt
from models.feedforward_nn import Model

model = Model.load("saved_models/model_four_cat_species.pickle")

class_names = ["caracal", "cheetah", "puma", "tiger"]

base_dir = "data/four cats photos/test"
folders = ["CARACAL", "CHEETAH", "PUMA", "TIGER"]

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

def predict_single_image(model, x):
    logits = model(x)
    probs = xp.exp(logits)
    probs = probs / xp.sum(probs, axis=1, keepdims=True)

    probs = probs[0]
    pred_idx = int(xp.argmax(probs))
    confidence = float(probs[pred_idx])

    return pred_idx, confidence

fig, axes = plt.subplots(len(folders), 4, figsize=(16, 4 * len(folders)))

for row, folder in enumerate(folders):
    folder_path = os.path.join(base_dir, folder)
    image_files = sorted(
    f for f in os.listdir(folder_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png")))[:4]


    for col, img_name in enumerate(image_files):
        path = os.path.join(folder_path, img_name)
        original_img, x = load_single_image(path)

        pred_idx, confidence = predict_single_image(model, x)
        pred_class = class_names[pred_idx]

        ax = axes[row, col]
        ax.imshow(original_img)
        ax.axis("off")
        ax.set_title(f"{pred_class}\n{confidence:.2f}", fontsize=10)

    axes[row, 0].set_ylabel(folder, fontsize=12, rotation=0, labelpad=40)

plt.tight_layout()
plt.show()

"""
Results saved at saved_models/four_cats_prediction_results.jpg
"""