# 🌊 Su Kütlesi Tespiti — Görüntü İşleme Tabanlı Proje

Bu proje, sabit görüntüler üzerinden su kütlelerinin tespit edilmesini hedefleyen, görüntü işleme temellerine dayalı bir çalışmadır. Başlangıç seviyesinde gri seviye görüntüler ve temel işlemlerle sınırlı olsa da, bu proje zamanla adım adım gelişerek gerçek zamanlı, drone destekli bir sistem haline gelmeyi amaçlamaktadır.

---

## 💡 Projenin Amacı

Başlangıçtaki temel hedefimiz, basit görüntüler üzerinden su alanlarını tespit etmek. Ancak uzun vadede bu proje, aşağıdaki gelişim adımlarını takip ederek **canlı video akışı üzerinde çalışan akıllı bir sistem** haline getirilecektir:

1. 📸 **Gri seviye görüntü işleme** (şu anki durum)
2. 🌈 **Renk uzayı analizi** (HSV, LAB gibi sistemlerle daha doğru segmentasyon)
3. 🤖 **Makine öğrenmesi ile desteklenmiş tespit** (özellik çıkarımı + sınıflandırma)
4. 🎥 **Canlı video üzerinden tespit** (kamera/dronelar ile)
5. 🛰️ **Gerçek zamanlı yer belirleme ve işaretleme**

Bu yapının sonunda amaç, örneğin bir drone ile uçarken, otonom bir şekilde su kütlelerini tanıyıp konumlarını işaretleyebilen bir sistem oluşturmaktır.

---

## 🧪 Kullanılan Yöntemler (Şu Ana Kadar)

- **OpenCV** – Görüntü işleme
- **Thresholding, Edge Detection, Morphology**
- **Python** – Tüm uygulama Python diliyle geliştirilmiştir

---

## 🚧 Gelecekteki Geliştirme Planı

- Renk tabanlı segmentasyon algoritmaları entegre edilecek
- Makineler için uygun öznitelik çıkarımı (alan, kontur yapısı vb.)
- Sınıflandırıcılarla su alanı/olmayan alan ayrımı
- Video işleme pipeline’ı hazırlanacak
- Gerçek zamanlı analiz için optimizasyon ve donanım bağlantıları (drone feed, GPS vs.)

---

## 🗂️ Proje Durumu

Şu anda proje sadece gri seviye görüntüler üzerinde çalışmakta ve temel segmentasyon algoritmaları uygulanmaktadır. Kod yapısı zamanla modüler hale getirilecek ve ileri seviye teknikler entegre edilecektir.

---

## ⚠️ Lisans

Bu proje sadece geliştirici olan **Ahmet** tarafından geliştirilmektedir. Kodun görüntülenmesine izin verilse de:

- Kullanılamaz
- Kopyalanamaz
- Dağıtılamaz
- Ticari veya akademik amaçla entegre edilemez

Detaylar için `LICENSE` dosyasına göz atabilirsiniz.

---

## 📌 Not

Proje sürekli gelişmektedir. Her adımda yapay zekâ ve görüntü işleme teknikleri ile daha yetenekli hale gelecektir.
