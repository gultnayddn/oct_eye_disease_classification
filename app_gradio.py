import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr
from functools import lru_cache

# ================== PATHS ==================
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data_raw", "OCT2017")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_PATH = os.path.join("models", "mobilenet_finetuned.keras")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("models", "mobilenet_stage1.keras")

# ================== DISEASE EXPLANATIONS ==================
DISEASE_INFO = {
    "CNV": {
        "title": "Koroidal Neovaskülarizasyon (CNV)",
        "desc": "Retina altında anormal damar oluşumu ile ilişkilidir. OCT görüntülerinde sıvı, düzensiz yansımalar veya yapısal bozulmalar görülebilir.",
        "risk": "Görme kaybı riski yüksek olabilir. Klinik değerlendirme önerilir."
    },
    "DME": {
        "title": "Diyabetik Makula Ödemi (DME)",
        "desc": "Diyabet kaynaklı makula bölgesinde sıvı birikimi/kalınlaşma görülebilir.",
        "risk": "Tedavi edilmezse merkezi görmeyi etkileyebilir. Klinik takip önemlidir."
    },
    "DRUSEN": {
        "title": "Drusen Birikimi",
        "desc": "Retina altında sarımsı birikimlerdir. OCT’de birikimlere bağlı kabarıklıklar/katman değişimleri görülebilir.",
        "risk": "Yaşa bağlı makula dejenerasyonu açısından risk göstergesi olabilir."
    },
    "NORMAL": {
        "title": "Normal Retina",
        "desc": "Retina katmanlarında belirgin patolojik bulgu tespit edilmemiş gibi görünür.",
        "risk": "Rutin takip önerilir."
    }
}

# ================== UTILS ==================
def get_class_names():
    return sorted([
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ])

CLASS_NAMES = get_class_names()

@lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

MODEL = load_model()

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img).astype("float32") / 255.0
    return np.expand_dims(x, axis=0)

def confidence_comment(conf: float):
    if conf >= 0.90:
        return "Model bu tahminde oldukça emin görünüyor."
    elif conf >= 0.70:
        return "Model orta-yüksek güvenle tahmin yaptı."
    else:
        return "Modelin güveni düşük; sınıflar arası benzerlik/karışma olabilir."

def predict_image(img):
    if img is None:
        return "Lütfen bir OCT görseli yükleyin (jpg/png).", {}

    x = preprocess(img)
    probs = MODEL.predict(x, verbose=0)[0]

    idx = int(np.argmax(probs))
    pred = CLASS_NAMES[idx]
    conf = float(probs[idx])

    info = DISEASE_INFO.get(pred, {"title": pred, "desc": "", "risk": ""})
    comment = confidence_comment(conf)

    # En çok karışan 2 sınıfı da yazalım (düşük güvende çok işe yarar)
    top_idx = np.argsort(probs)[::-1][:2]
    alt = [(CLASS_NAMES[int(i)], float(probs[int(i)])) for i in top_idx]

    text = (
        f"Model: {os.path.basename(MODEL_PATH)}\n\n"
        f"Tahmin: {pred} — {info['title']}\n"
        f"Güven: {conf:.4f}\n\n"
        f"Açıklama:\n{info['desc']}\n\n"
        f"Klinik Değerlendirme:\n{info['risk']}\n\n"
        f"Model Yorumu:\n{comment}\n\n"
        f"En olası 2 sınıf:\n"
        f"- {alt[0][0]}: {alt[0][1]:.4f}\n"
        f"- {alt[1][0]}: {alt[1][1]:.4f}\n\n"
        f"Not: Bu sistem bir karar destek aracıdır; klinik tanı yerine geçmez."
    )

    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return text, prob_dict

def random_image(seed=42):
    random.seed(int(seed))

    imgs = []
    for root, _, files in os.walk(TEST_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append(os.path.join(root, f))

    path = random.choice(imgs)
    img = Image.open(path).convert("RGB")
    return img, f"Seçilen dosya:\n{path}"

# ================== GRADIO UI ==================
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 OCT Göz Hastalığı Sınıflandırma (Gradio)")
    gr.Markdown("**CNN + MobileNetV2 + Transfer Learning**  \nSınıflar: CNV / DME / DRUSEN / NORMAL")

    with gr.Row():
        img_input = gr.Image(type="pil", label="OCT Görüntüsü Yükle")
        txt_output = gr.Textbox(label="Açıklamalı Sonuç", lines=16)

    prob_output = gr.Label(num_top_classes=4, label="Sınıf Olasılıkları (Top-4)")

    btn_predict = gr.Button("🔍 Tahmin Et")
    btn_predict.click(
        fn=predict_image,
        inputs=img_input,
        outputs=[txt_output, prob_output]
    )

    gr.Markdown("## 🎲 Rastgele Test Görüntüsü ile Deneme")
    seed_input = gr.Number(value=42, label="Seed", precision=0)
    btn_random = gr.Button("Random Görsel Seç")
    info_box = gr.Textbox(label="Bilgi", lines=2)

    btn_random.click(
        fn=random_image,
        inputs=seed_input,
        outputs=[img_input, info_box]
    )

    gr.Markdown(f"**Not:** Finetuned model yoksa otomatik olarak stage1 kullanılır. Şu an kullanılan: `{os.path.basename(MODEL_PATH)}`")

if __name__ == "__main__":
    demo.launch()

