import os
import tensorflow as tf

# ============ AYARLAR ============
DATA_DIR = os.path.join(os.getcwd(), "data_raw", "OCT2017")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# HIZLI DENEME: eğitim süresini kısaltmak için
MAX_TRAIN_BATCHES = 50   # 200 batch sonra epoch bitsin (sonra arttıracağız)
MAX_VAL_BATCHES   = 20

# ============ DATASET ============
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Normalize + performans
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).take(MAX_TRAIN_BATCHES).prefetch(AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).take(MAX_VAL_BATCHES).prefetch(AUTOTUNE)

# ============ MODEL (MobileNetV2) ============
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False  # önce donduruyoruz

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)  # mobilenet preprocess
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============ TRAIN ============
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2
)

# Kaydet
os.makedirs("models", exist_ok=True)
model.save("models/mobilenet_stage1.keras")
print("SAVED models/mobilenet_stage1.keras")
