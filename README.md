# OCT Görüntülerinden Retina Hastalıklarının Derin Öğrenme ile Sınıflandırılması

Bu proje, Optik Koherens Tomografi (OCT) görüntülerinden retina hastalıklarının otomatik olarak sınıflandırılmasını amaçlayan, Konvolüsyonel Sinir Ağları (CNN) tabanlı bir derin öğrenme uygulamasıdır. Proje, bilgisayar destekli bir karar destek sistemi olarak tasarlanmıştır.

---

## 📌 Proje Konusu ve Seçilme Gerekçesi

Retina hastalıkları (CNV, DME, DRUSEN) erken teşhis edilmediği takdirde kalıcı görme kaybına yol açabilmektedir. OCT görüntüleme, göz hastalıklarının teşhisinde yaygın olarak kullanılan, invaziv olmayan bir yöntemdir.

Bu projede, OCT görüntülerinin manuel olarak incelenmesi yerine, derin öğrenme tabanlı bir sistem geliştirilerek:
- Tanı sürecinin hızlandırılması
- Uzmanlara destek olunması
- Hata oranlarının azaltılması

amaçlanmıştır.

---

## 📚 Literatür Özeti (İlgili Çalışmalar)

Literatürde OCT görüntülerinin sınıflandırılması için sıklıkla CNN tabanlı mimariler kullanılmaktadır. VGG, ResNet ve DenseNet gibi modeller yüksek doğruluk sağlasa da hesaplama maliyetleri yüksektir.

Bu projede **MobileNetV2** mimarisi tercih edilmiştir çünkü:
- Daha az parametreye sahiptir
- Daha hızlıdır
- CPU üzerinde çalışmaya daha uygundur
- Yeterli doğruluk sağlar

Bu yönleriyle gerçek dünya uygulamaları için daha verimli bir yaklaşımdır.

---

## 📊 Veri Seti

- **Veri Seti Adı:** OCT2017
- **Kaynak:** Kaggle
- **Toplam Görüntü Sayısı:** ~108.000
- **Sınıflar:**
  - CNV (Choroidal Neovascularization)
  - DME (Diabetic Macular Edema)
  - DRUSEN
  - NORMAL

Veri seti eğitim, doğrulama ve test olarak ayrılmıştır. Veri setinin boyutunun büyük olması nedeniyle GitHub reposuna eklenmemiştir.

---

## ⚙️ Kullanılan Yöntem ve Yaklaşım

- Konvolüsyonel Sinir Ağları (CNN)
- Transfer Learning (MobileNetV2)
- TensorFlow / Keras
- Görüntü ön işleme ve normalizasyon
- Veri artırma (Data Augmentation)
- İnce ayar (Fine-tuning)

---

## 🏋️ Model Eğitimi

- Giriş boyutu: 224 × 224 RGB
- Optimizasyon algoritması: Adam
- Kayıp fonksiyonu: Categorical Cross-Entropy
- Başarı metriği: Accuracy
- Eğitim ortamı: CPU

---

## 📈 Model Değerlendirme

Model performansı aşağıdaki metrikler kullanılarak değerlendirilmiştir:
- Doğruluk (Accuracy)
- Precision, Recall, F1-Score
- Confusion Matrix

Elde edilen sonuçlar, modelin genel olarak başarılı olduğunu ancak bazı sınıflar (özellikle DRUSEN) arasında görsel benzerlik nedeniyle karışmalar yaşandığını göstermektedir.

---

## 🖥️ Uygulama ve Arayüz (Gradio)

Modelin test edilebilmesi için Gradio tabanlı bir web arayüzü geliştirilmiştir. Bu arayüz sayesinde:
- Rastgele veya kullanıcı tarafından seçilen bir OCT görüntüsü sınıflandırılabilir
- Tahmin edilen sınıf ve güven skoru görüntülenir
- Klinik yorum benzeri açıklamalar sunulur

---

## ⚠️ Uyarı

Bu proje **akademik amaçlıdır**.  
Geliştirilen sistem bir **karar destek aracıdır** ve **klinik tanı yerine geçmez**.

---

## 🚀 Çalıştırma Adımları

```bash
pip install -r requirements.txt
python app_gradio.py
