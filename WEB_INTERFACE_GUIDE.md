# Mulholland Drive RAG Chatbot - Web Arayüzü & Product Kılavuzu

## 🎯 Proje Özeti

Bu proje, **Mulholland Drive** filmi hakkında bilgi tabanına dayalı (RAG) bir sohbet deneyimi sunar. Kullanıcılar filmin çok katmanlı yapısı, psikanalitik motifleri ve karakter dinamikleri hakkında sorular sorabilir ve bağlama dayalı, tutarlı yanıtlar alabilir.

## 🏗️ Teknik Mimari

### RAG Pipeline Akışı
```
Kullanıcı Sorusu → Embedding → Vektör Arama → Bağlam Toplama → Gemini LLM → Yanıt
```

1. **Retrieval**: Kullanıcı sorusu vektörleştirilir ve `chunks.jsonl` içindeki en benzer parçalar bulunur
2. **Augmentation**: Bulunan bağlam, sistem prompt'u ile birleştirilir
3. **Generation**: Gemini LLM, bağlama dayalı yanıt üretir

### Kullanılan Teknolojiler
- **Frontend**: Streamlit (Python web framework)
- **Embedding**: LangChain + HuggingFace SentenceTransformers
- **LLM**: Google Gemini 1.5 Pro (google-generativeai)
- **Vektör Arama**: In-memory cosine similarity
- **Veri Formatı**: JSONL (chunks + embeddings)

## 🚀 Deployment Seçenekleri

### 1. Streamlit Community Cloud (Önerilen)

**Avantajlar:**
- Ücretsiz hosting
- Otomatik GitHub entegrasyonu
- Kolay secrets yönetimi
- Hızlı deployment

**Adımlar:**
1. GitHub'a repo yükle
2. [share.streamlit.io](https://share.streamlit.io) → "New app"
3. Repo'yu seç, `rag_chatbot.py` dosyasını belirt
4. Secrets'e `GOOGLE_API_KEY` ekle
5. Deploy et

**URL Formatı:** `https://KULLANICI_ADI-mulholland-rag-chatbot-app-xyz123.streamlit.app`

### 2. Hugging Face Spaces

**Avantajlar:**
- AI/ML odaklı platform
- Kolay model paylaşımı
- Community features
- GPU desteği (ücretli)

**Adımlar:**
1. [huggingface.co/spaces](https://huggingface.co/spaces) → "Create new Space"
2. Space adı: `mulholland-rag-chatbot`
3. SDK: Streamlit seç
4. Dosyaları yükle
5. Secrets'e API key ekle

**URL Formatı:** `https://huggingface.co/spaces/KULLANICI_ADI/mulholland-rag-chatbot`

### 3. Diğer Seçenekler

- **Heroku**: Ücretli, kolay deployment
- **Railway**: Modern alternatif
- **Google Cloud Run**: Serverless, ölçeklenebilir
- **AWS EC2**: Tam kontrol, ücretli

## 🧪 Test Senaryoları

### Temel Fonksiyonellik Testleri

#### 1. Film Özeti ve Genel Bilgi
```
Soru: "Mulholland Drive'ın konusu nedir?"
Beklenen: Filmin temel hikayesi, karakterler, ana temalar
```

#### 2. Karakter Analizi
```
Soru: "Betty ve Diane arasındaki ilişki nasıl açıklanabilir?"
Beklenen: Kimlik bölünmesi, psikanalitik açıklama, rüya/gerçek geçişi
```

#### 3. Psikanalitik Semboller
```
Soru: "Filmin psikanalitik sembolleri nelerdir?"
Beklenen: Mavi kutu, Club Silencio, rüya sekansları, bastırma mekanizmaları
```

#### 4. Motif ve Tema Analizi
```
Soru: "Mavi kutu ve mavi anahtar neyi temsil ediyor?"
Beklenen: Arzu, erişilemeyen gerçeklik, psikanalitik sembolizm
```

#### 5. Yapısal Analiz
```
Soru: "Filmin rüya ve gerçek arasındaki geçişleri nasıl yorumlanabilir?"
Beklenen: Lynch'in sinematik teknikleri, bilinçdışı temsili
```

### Performans Testleri

#### Yanıt Kalitesi
- **Tutarlılık**: Aynı soruya benzer yanıtlar
- **Doğruluk**: Bağlama dayalı, varsayım yapmayan
- **Derinlik**: Psikanalitik çerçevede analiz
- **Uzunluk**: Yeterince detaylı ama özlü

#### Teknik Performans
- **Yanıt Süresi**: < 10 saniye
- **Retrieval Kalitesi**: Top-K parçaların ilgili olması
- **API Limitleri**: Gemini quota aşımı kontrolü
- **Hata Yönetimi**: Graceful degradation

### Edge Case Testleri

#### 1. Belirsiz Sorular
```
"Mulholland Drive hakkında ne biliyorsun?"
→ Genel bilgi, örnek sorular önerme
```

#### 2. Çok Spesifik Sorular
```
"Club Silencio sahnesinde kaç kişi var?"
→ Bağlama dayalı yanıt veya "bilgi tabanında bulunamadı"
```

#### 3. Film Dışı Sorular
```
"David Lynch'in diğer filmleri nelerdir?"
→ "Bu uygulama sadece Mulholland Drive hakkında" mesajı
```

## 📊 Kullanıcı Deneyimi (UX)

### Arayüz Özellikleri

#### Ana Sayfa
- **Başlık**: Mulholland Drive RAG Chatbot
- **Açıklama**: Proje hakkında kısa bilgi
- **Örnek Sorular**: Hızlı başlangıç için
- **Soru Girişi**: Büyük text area
- **Sor Butonu**: Prominent, primary style

#### Sidebar Ayarları
- **Chunks Dosyası Yolu**: Esnek dosya konumu
- **Top-K Sayısı**: Retrieval kalitesi kontrolü
- **API Key Durumu**: Görsel feedback
- **Yükleme Durumu**: Progress indicators

#### Yanıt Alanı
- **Ana Yanıt**: Temiz, okunabilir format
- **Kaynak Parçalar**: Expandable, skorlu
- **Loading States**: Spinner ve progress
- **Hata Mesajları**: Açık, çözüm odaklı

### Kullanıcı Yolculuğu (User Journey)

#### 1. İlk Ziyaret
```
Landing → Örnek Soru Seçimi → Soru Yazma → Yanıt Alma → Kaynak İnceleme
```

#### 2. Tekrar Ziyaret
```
Direkt Soru Yazma → Hızlı Yanıt → Detaylı Analiz → Yeni Soru
```

#### 3. Derinlemesine Kullanım
```
Spesifik Soru → Kaynak İnceleme → Follow-up Sorular → Karşılaştırmalı Analiz
```

## 🔧 Bakım ve Güncelleme

### Düzenli Kontroller

#### 1. API Limitleri
- Gemini quota kullanımı
- Rate limiting kontrolü
- Error rate monitoring

#### 2. Embedding Kalitesi
- Retrieval accuracy
- Chunk relevance scores
- Model performance

#### 3. Kullanıcı Geri Bildirimi
- Yanıt kalitesi değerlendirmesi
- Hata raporları
- Feature requests

### Güncelleme Stratejileri

#### 1. Model Güncellemeleri
- Yeni embedding modelleri
- Gemini model versiyonları
- Performance optimizasyonları

#### 2. Veri Güncellemeleri
- Yeni film analizi ekleme
- Chunk kalitesi iyileştirme
- Metadata zenginleştirme

#### 3. Arayüz İyileştirmeleri
- UX/UI güncellemeleri
- Yeni özellikler
- Performance optimizasyonları

## 📈 Metrikler ve Analitik

### Temel Metrikler
- **Günlük Aktif Kullanıcı**: DAU
- **Soru Sayısı**: Günlük/toplam
- **Yanıt Süresi**: Ortalama, medyan
- **Başarı Oranı**: Hata-free yanıtlar

### Kalite Metrikleri
- **Retrieval Relevance**: Top-K skorları
- **Yanıt Uzunluğu**: Token sayısı
- **Kullanıcı Memnuniyeti**: Feedback skorları
- **API Kullanımı**: Cost per query

### Teknik Metrikler
- **Uptime**: %99+ hedef
- **Response Time**: <10s hedef
- **Error Rate**: <5% hedef
- **Memory Usage**: Optimizasyon

## 🎯 Gelecek Geliştirmeler

### Kısa Vadeli (1-3 ay)
- **Chat History**: Konuşma geçmişi
- **Export Features**: Yanıtları kaydetme
- **Mobile Optimization**: Responsive design
- **Multi-language**: İngilizce/Türkçe

### Orta Vadeli (3-6 ay)
- **Advanced RAG**: Re-ranking, hybrid search
- **Visual Elements**: Film stilleri, diyagramlar
- **Comparison Mode**: Karakter/film karşılaştırması
- **API Endpoints**: Programmatic access

### Uzun Vadeli (6+ ay)
- **Multi-film Support**: Lynch filmleri
- **Interactive Analysis**: Kullanıcı katılımlı analiz
- **AI-powered Insights**: Otomatik tema keşfi
- **Community Features**: Kullanıcı yorumları

## 📞 Destek ve İletişim

### Teknik Destek
- **GitHub Issues**: Bug reports, feature requests
- **Documentation**: README, API docs
- **Community**: Discord/Slack channels

### Kullanıcı Destek
- **FAQ**: Sık sorulan sorular
- **Tutorials**: Video rehberler
- **Contact**: Email support

---

Bu kılavuz, Mulholland Drive RAG Chatbot'un web arayüzü ve product yönetimi için kapsamlı bir rehberdir. Proje geliştikçe güncellenmelidir.
