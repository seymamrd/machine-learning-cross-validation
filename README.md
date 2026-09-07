# 🤖 Machine Learning: Logistic Regression & GridSearchCV Optimization

Bu proje, **Lojistik Regresyon** algoritması üzerinde hiperparametre optimizasyonu ve Model Doğrulama (Cross-Validation) süreçlerini deneysel olarak incelemek amacıyla hazırlanmıştır. 

Projede temel hedef; aşırı öğrenmeyi (overfitting) engellemek, düzenlileştirme (regularization) cezalarını (`L1` vs `L2`) kıyaslamak ve model performansını sistematik bir şekilde artırmaktır.

---

## 📌 Öne Çıkan Konseptler ve Teknik Detaylar

* **Lojistik Regresyon:** Kategorik verileri sınıflandırmak için Sigmoid fonksiyonu üzerinden olasılıksal tahminler üretilmiştir.
* **Hiperparametre Optimizasyonu (`GridSearchCV`):** Model başarısını en üst seviyeye çıkarmak için farklı parametre kombinasyonları taranmıştır.
* **Arama Aralıkları (`np.logspace` vs `np.linspace`):** 
  * `np.logspace`: Ceza katsayısı ($C$) üzerinde geniş aralıklı (üstel/katlanarak büyüyen) arama yapmak için kullanılmıştır.
  * `np.linspace`: Kazanan $C$ değeri etrafında nokta atışı yapabilmek adına doğrusal aralıkta ince ayar (fine-tuning) amaçlı kurgulanmıştır.
* **Düzenlileştirme (Regularization):**
  * `L1 (Lasso)`: Etkisiz özniteliklerin katsayılarını sıfırlayarak öznitelik seçimi sağlamıştır.
  * `L2 (Ridge)`: Katsayıları küçülterek katsayı patlamasını önlemiştir.
* **Optimizasyon Motoru (`SAGA`):** `L1` cezasını ve büyük veri yapılarını destekleyen türevsel hesaplama algoritması tercih edilmiştir.

---

## 🛠️ Kullanılan Teknolojiler

* **Python 3.x**
* **NumPy & Pandas:** Veri işleme ve matris operasyonları
* **Scikit-Learn:** `LogisticRegression`, `GridSearchCV`, `confusion_matrix`
* **Matplotlib & Seaborn:** Hata matrisi ve grafik görselleştirme

---

## 📊 Proje Akışı (Pipeline)

1. **Veri Ön İşleme:** Sınıflandırma verisinin hazırlanması ve ölçeklenmesi.
2. **Grid Oluşturma:**
   ```python
   grid = {
       "C": np.logspace(-3, 3, 7), 
       "penalty": ["l1", "l2"], 
       "solver": ["saga"]
   }