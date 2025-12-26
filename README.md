# OCT Görüntülerinden Retina Hastalıklarının Derin Öğrenme ile Sınıflandırılması

Bu proje, Optik Koherens Tomografi (OCT) görüntülerinden retina hastalıklarının otomatik olarak sınıflandırılmasını amaçlayan, Konvolüsyonel Sinir Ağları (CNN) tabanlı bir derin öğrenme uygulamasıdır. Proje, bilgisayar destekli bir karar destek sistemi olarak tasarlanmıştır.

---

## 📌 Proje Konusu ve Seçilme Gerekçesi

Bu projenin konusu, Optik Koherens Tomografi (OCT) görüntüleri kullanılarak retina hastalıklarının derin öğrenme tabanlı Konvolüsyonel Sinir Ağları (CNN) ile otomatik olarak sınıflandırılmasıdır.

Retina hastalıkları, dünya genelinde görme kaybının en yaygın nedenleri arasında yer almaktadır. Özellikle Koroidal Neovaskülarizasyon (CNV), Diyabetik Makula Ödemi (DME) ve Drusen gibi hastalıklar erken teşhis edilmediğinde kalıcı görme kaybına yol açabilmektedir. OCT görüntüleme yöntemi, retina tabakalarının yüksek çözünürlüklü kesitsel görüntülerini sağlayarak bu hastalıkların teşhisinde önemli bir rol oynamaktadır.

Bu proje, hem sağlık alanında gerçek bir probleme çözüm sunması, hem de CNN tabanlı görüntü işleme tekniklerinin uygulanabilirliğini göstermesi açısından seçilmiştir. Ayrıca proje, derin öğrenmenin teorik bilgisinin pratik bir uygulama ile pekiştirilmesini hedeflemektedir.

---

## 📚 Literatür Özeti (İlgili Çalışmalar)

Son yıllarda derin öğrenme yöntemleri, tıbbi görüntü analizi alanında yaygın olarak kullanılmaktadır. Literatürde OCT görüntülerinden retina hastalıklarının tespiti için VGG, ResNet, Inception ve MobileNet gibi CNN mimarilerinin başarıyla uygulandığı birçok çalışma bulunmaktadır.

Bu çalışmalar, derin öğrenme modellerinin uzman hekimlere yardımcı olabilecek karar destek sistemleri olarak kullanılabileceğini göstermektedir. Otomatik sistemler, özellikle yoğun hasta sayısına sahip sağlık kuruluşlarında erken teşhis sürecini hızlandırarak klinik iş yükünü azaltabilir.

Bu bağlamda, seçilen konu hem akademik olarak güncel, hem de toplumsal ve klinik açıdan yüksek öneme sahip bir problem alanını kapsamaktadır.

---

## 📊 Veri Seti

Projede kullanılan veri seti, Kaggle platformunda yayımlanan OCT2017 veri setidir. Bu veri seti, binlerce gerçek hasta OCT görüntüsünden oluşmaktadır ve literatürde yaygın olarak kullanılmaktadır.

Veri Seti Özellikleri:

- Toplam 80.000+ eğitim görüntüsü

4 sınıf:

- CNV (Koroidal Neovaskülarizasyon)

- DME (Diyabetik Makula Ödemi)

- DRUSEN

- NORMAL

Ayrılmış train / validation / test klasör yapısı

Veri setinin büyük olması, derin öğrenme modellerinin genelleme yeteneğini artırmakta ve projenin akademik güvenilirliğini yükseltmektedir.

---

## ⚙️ Kullanılan Yöntem ve Yaklaşım

Bu projede Konvolüsyonel Sinir Ağları (CNN) kullanılmıştır. CNN’ler, görüntülerdeki kenar, doku ve şekil gibi uzamsal özellikleri otomatik olarak öğrenebilme yeteneğine sahiptir.

Neden MobileNetV2?

Literatürde sık kullanılan VGG ve ResNet gibi modeller yüksek doğruluk sunsa da, bu modeller:

Daha fazla parametre içerir

Daha yüksek hesaplama maliyeti gerektirir

Bu projede ise:

Daha hafif

Daha hızlı

CPU üzerinde çalışabilir

bir mimari tercih edilmiştir. Bu nedenle MobileNetV2 seçilmiştir.

---

## 🏋️ Model Eğitimi

Model eğitimi aşağıdaki adımlarla gerçekleştirilmiştir:

Görüntüler 224×224 boyutuna yeniden ölçeklendirilmiştir

Piksel değerleri [0,1] aralığına normalize edilmiştir

MobileNetV2 tabanı dondurulmuş (freeze) şekilde kullanılmıştır

Üstüne özel sınıflandırıcı katmanlar eklenmiştir

Categorical Cross-Entropy kayıp fonksiyonu

Adam optimizasyon algoritması kullanılmıştır

Eğitim süreci sırasında doğruluk (accuracy) ve kayıp (loss) değerleri izlenmiştir.

---

## 📈 Model Değerlendirme

Model performansı, daha önce hiç görülmemiş test veri seti üzerinde değerlendirilmiştir.

Elde Edilen Sonuçlar:

Test Accuracy: 0.7097

Macro F1 Score: 0.6843

Weighted F1 Score: 0.6843

Ayrıca sınıf bazlı performans değerlendirmesi için:

Confusion Matrix

Precision / Recall / F1-score

hesaplanmış ve görselleştirilmiştir.

Sonuçlar, modelin genel olarak sınıfları ayırt edebildiğini ancak bazı sınıflar arasında (özellikle benzer patolojik yapılar içeren sınıflarda) karışmalar yaşanabildiğini göstermektedir.

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
## Genel Değerlendirme ve Sonuç

Bu proje kapsamında, derin öğrenme teknikleri kullanılarak retina hastalıklarının otomatik sınıflandırılması başarıyla gerçekleştirilmiştir. Elde edilen sonuçlar, CNN tabanlı modellerin tıbbi görüntü analizinde etkili bir şekilde kullanılabileceğini göstermektedir.

Proje, hem teorik bilgilerin pratiğe dökülmesi hem de gerçek bir problem üzerinde çalışılması açısından önemli kazanımlar sağlamıştır. Gelecek çalışmalarda:

Daha ileri fine-tuning

Farklı mimarilerin karşılaştırılması

Daha kapsamlı klinik veri kullanımı

ile model performansının daha da artırılabileceği düşünülmektedir.
 

## 🚀 Çalıştırma Adımları

```bash
pip install -r requirements.txt
python app_gradio.py
