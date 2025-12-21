import os
import random
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

def list_images(root_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(dirpath, fn))
    return files

def get_class_names(train_dir: str):
    # Keras alfabetik sıralıyor
    return sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

def load_image(path, img_size=(224, 224)):
    img = Image.open(path).convert("RGB").resize(img_size)
    x = np.array(img).astype("float32") / 255.0
    return np.expand_dims(x, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join("data_raw", "OCT2017"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model", default="models/mobilenet_finetuned.keras")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None, help="Aynı sonucu görmek için bir sayı ver (örn 42)")
    args = parser.parse_args()

    # Model seçimi (finetuned yoksa stage1)
    model_path = args.model
    if not os.path.exists(model_path):
        fallback = "models/mobilenet_stage1.keras"
        if os.path.exists(fallback):
            print(f"[INFO] '{model_path}' yok. '{fallback}' kullanılacak.")
            model_path = fallback
        else:
            raise FileNotFoundError("Model bulunamadı. Önce eğitimi çalıştırıp models/*.keras üretmelisin.")

    split_dir = os.path.join(args.data_dir, args.split)
    train_dir = os.path.join(args.data_dir, "train")
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split klasörü yok: {split_dir}")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train klasörü yok: {train_dir}")

    if args.seed is not None:
        random.seed(args.seed)

    files = list_images(split_dir)
    if not files:
        raise RuntimeError(f"{split_dir} içinde görsel bulunamadı.")

    picked = random.choice(files)

    class_names = get_class_names(train_dir)
    model = tf.keras.models.load_model(model_path)

    x = load_image(picked, (224, 224))
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]

    print("\n=== RANDOM IMAGE PICKED ===")
    print(picked)

    print("\n=== PREDICTION ===")
    print("Model:", model_path)
    print("Pred :", pred_name)
    print("Prob :", float(probs[pred_idx]))

    k = min(args.topk, len(class_names))
    top_idx = np.argsort(probs)[::-1][:k]
    print("\n=== TOP-K ===")
    for i in top_idx:
        print(f"{class_names[int(i)]:<10}  {float(probs[int(i)]):.4f}")

    # Basit yorum
    p = float(probs[pred_idx])
    if p >= 0.90:
        comment = "Model bu tahminde oldukça emin görünüyor."
    elif p >= 0.70:
        comment = "Model orta-yüksek güvenle tahmin yaptı."
    else:
        comment = "Modelin güveni düşük; sınıflar arası benzerlik/karışma olabilir."
    print("\n=== COMMENT ===")
    print(comment)

if __name__ == "__main__":
    main()
