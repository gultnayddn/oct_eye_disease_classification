import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = os.path.join(os.getcwd(), "data_raw", "OCT2017")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# Test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
print("Classes:", class_names)

# Normalize
test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

# Model yükle
model = tf.keras.models.load_model("models/mobilenet_stage1.keras")

# Tahmin
y_true = []
y_pred = []

for batch_x, batch_y in test_ds:
    probs = model.predict(batch_x, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true.extend(batch_y.numpy().tolist())
    y_pred.extend(preds.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Rapor
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Confusion matrix'i kaydet (görsel)
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix_stage1.png", dpi=200)
print("\nSaved: outputs/confusion_matrix_stage1.png")
