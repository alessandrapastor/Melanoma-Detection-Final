import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox

# ------------------ 1. Class mapping ------------------
class_mapping = {
    'nv':   'Melanocytic Nevi',
    'mel':  'Melanoma',
    'bkl':  'Benign Keratosis',
    'bcc':  'Basal Cell Carcinoma',
    'akiec':'Actinic Keratoses',
    'vasc': 'Vascular Lesions',
    'df':   'Dermatofibroma'
}

# ------------------ 2. Load model & encoder ------------------
MODEL_PATH = "skin_cancer_model.h5"
LE_PATH    = "label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)

# Determine how many tabular features the model expects
TAB_FEATURE_DIM = int(model.inputs[1].shape[1])

# ------------------ 2a. Define possible locations ------------------
possible_locations = [
    "Arm", "Leg", "Back", "Chest", "Face", "Scalp", "Abdomen", "Neck"
]

# ------------------ 3. Grad-CAM utils ------------------
def get_gradcam_heatmap(img_array, tab_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output,
         model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, tab_array])
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ tf.expand_dims(pooled_grads, axis=-1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def superimpose_heatmap(orig_img, heatmap, alpha=0.4):
    hmap_uint8 = np.uint8(255 * heatmap)
    hmap_color = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)
    hmap_color = cv2.resize(hmap_color, (orig_img.shape[1], orig_img.shape[0]))
    return cv2.addWeighted(orig_img, 1 - alpha, hmap_color, alpha, 0)

# ------------------ 4. Image + prediction pipeline ------------------
def process_image(image_path, age, location):
    # true label from filename
    short_true = os.path.basename(image_path).split('_')[0]
    true_label = class_mapping.get(short_true, short_true)

    # load & preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image at {image_path}")
    img_resized = cv2.resize(img, (224, 224))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input   = np.expand_dims(img_resized, axis=0)

    # build tabular input: [age, one-hot-location, … zeros if extra dims]
    tab_input = np.zeros((1, TAB_FEATURE_DIM), dtype=np.float32)
    try:
        age_val = float(age)
    except ValueError:
        raise ValueError("Age must be a number.")
    # simple normalization (optional—you can adjust as needed)
    tab_input[0,0] = age_val / 100.0

    # one-hot encode location
    if location not in possible_locations:
        raise ValueError(f"Location must be one of {possible_locations}")
    loc_idx = possible_locations.index(location)
    # put it at index 1 + loc_idx, if space allows
    if 1 + loc_idx < TAB_FEATURE_DIM:
        tab_input[0, 1 + loc_idx] = 1.0

    # predict
    preds = model.predict([img_input, tab_input])[0]
    top_idxs  = np.argsort(preds)[::-1][:2]
    top_confs = preds[top_idxs] * 100  # as percentages
    short_preds = le.inverse_transform(top_idxs)
    top_labels  = [ class_mapping.get(s, s) for s in short_preds ]

    # Grad-CAM on the top-1
    last_conv = "conv2d_5"   # adjust to your model’s last‐conv layer name
    heatmap   = get_gradcam_heatmap(img_input, tab_input, model, last_conv, pred_index=top_idxs[0])
    overlay   = superimpose_heatmap(img_rgb, heatmap)

    return true_label, top_labels, top_confs, img_rgb, overlay

# ------------------ 5. GUI callback ------------------
def on_select_image():
    path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Images","*.jpg;*.jpeg;*.png"),("All files","*.*")]
    )
    if not path:
        return
    age = age_entry.get().strip()
    location = location_var.get()

    try:
        true_lbl, pred_labels, confs, orig, cam = process_image(path, age, location)

        plt.figure(figsize=(10,5))
        # original
        plt.subplot(1,2,1)
        plt.imshow(orig)
        plt.title(f"Original\nTrue: {true_lbl}\nAge: {age}, Loc: {location}")
        plt.axis('off')

        # grad-cam + top-2 preds
        plt.subplot(1,2,2)
        plt.imshow(cam)
        title = "Grad-CAM\n"
        title += f"1: {pred_labels[0]} ({confs[0]:.1f}%)\n"
        title += f"2: {pred_labels[1]} ({confs[1]:.1f}%)"
        plt.title(title)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ------------------ 6. Build the window (centered + larger) ------------------
root = tk.Tk()
root.title("Grad-CAM Viewer with Age & Location")

# desired window size
win_w, win_h = 800, 600

# update idle tasks to get correct screen size
root.update_idletasks()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()

# compute coordinates to center the window
x = (screen_w // 2) - (win_w // 2)
y = (screen_h // 2) - (win_h // 2)

# set geometry: width x height + x_offset + y_offset
root.geometry(f"{win_w}x{win_h}+{x}+{y}")

# rest of your widgets…
# Age input
tk.Label(root, text="Patient Age (years):").pack(anchor="w", padx=20, pady=(20,0))
age_entry = tk.Entry(root)
age_entry.pack(fill="x", padx=20)

# Location dropdown
tk.Label(root, text="Lesion Location:").pack(anchor="w", padx=20, pady=(10,0))
location_var = tk.StringVar(value=possible_locations[0])
tk.OptionMenu(root, location_var, *possible_locations).pack(fill="x", padx=20)

# Show full list of locations
tk.Label(root,
         text="Possible locations: " + ", ".join(possible_locations),
         wraplength=760,
         fg="gray"
        ).pack(anchor="w", padx=20, pady=(5,10))

# Select Image button
btn = tk.Button(root,
                text="Select Image & Predict",
                command=on_select_image,
                width=30, height=2)
btn.pack(pady=(0,30))

root.mainloop()
