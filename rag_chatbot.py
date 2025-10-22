#!/usr/bin/env python3
"""
Mulholland Drive RAG Chatbot - Streamlit Web Interface

Bu uygulama, Mulholland Drive filmi hakkında bilgi tabanına dayalı (RAG) bir
sohbet deneyimi sağlar. RAG akışı:

1) Retrieval: Kullanıcı sorusuna en ilgili metin parçalarını bulmak için
   vektör benzerliği (cosine similarity) kullanılır.
2) Augmented Prompt: Bulunan bağlam, kullanıcı sorusuyla birlikte LLM'e gönderilir.
3) Generation: LLM, filmin çok katmanlı yapısını bağlama dayalı açıklar.

Bu mimari, filmin belirsiz yapısını "belgeye dayalı ipuçları" ile sabitlemeyi amaçlar.
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import streamlit as st
from dotenv import load_dotenv

# .env dosyasını otomatik yükle
load_dotenv()


@st.cache_data
def load_embeddings_and_chunks(path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    chunks.jsonl içinden embedding vektörlerini ve ilgili metin parçalarını yükler.
    Cache'lenir, böylece tekrar yüklenmez.
    """
    try:
        vectors: List[List[float]] = []
        chunks: List[Dict[str, Any]] = []
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    rec = json.loads(line.strip())
                    if "embedding" in rec and "text" in rec:
                        vectors.append(rec["embedding"])
                        chunks.append(rec)
                except json.JSONDecodeError as e:
                    st.error(f"JSON hatası, satır {line_num}: {e}")
                    continue
        
        if not vectors:
            raise ValueError("chunks.jsonl içinde geçerli embedding bulunamadı.")
        
        emb = np.array(vectors, dtype=np.float32)
        return emb, chunks
        
    except FileNotFoundError:
        raise ValueError(f"Dosya bulunamadı: {path}")
    except Exception as e:
        raise ValueError(f"Dosya yüklenirken hata: {e}")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """Vektörleri normalize et (cosine similarity için)"""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


@st.cache_resource
def get_embedder():
    """Embedding modelini cache'le"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            return None


def embed_query(text: str) -> Optional[np.ndarray]:
    """Sorguyu vektörleştir"""
    embedder = get_embedder()
    if embedder is None:
        return None
    
    try:
        vec = embedder.embed_query(text)
        return np.array(vec, dtype=np.float32)
    except Exception as e:
        return None


def retrieve_top_k(
    query_vec: np.ndarray,
    doc_embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Cosine similarity ile en benzer top_k parçayı döndür"""
    if query_vec is None or doc_embeddings.size == 0:
        return []
    
    q = query_vec.reshape(1, -1)
    q = normalize_rows(q)
    d = normalize_rows(doc_embeddings)
    
    sims = (q @ d.T).flatten()
    idx = np.argsort(-sims)[:top_k]
    
    results: List[Dict[str, Any]] = []
    for i in idx:
        item = chunks[int(i)].copy()
        item["score"] = float(sims[int(i)])
        results.append(item)
    
    return results


def call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Google Gemini ile yanıt üret"""
    try:
        import google.generativeai as genai
    except ImportError:
        return "Sistem şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "Sistem şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"{system_prompt}\n\nKullanıcı sorusu:\n{user_prompt}"
        resp = model.generate_content(prompt)
        
        if hasattr(resp, "text") and resp.text:
            return resp.text
        
        # Alternatif çıktı formatı
        try:
            return resp.candidates[0].content.parts[0].text
        except (AttributeError, IndexError):
            return "Analiz tamamlanamadı. Lütfen sorunuzu farklı şekilde ifade edin."
            
    except Exception as e:
        return "Üzgünüm, bu konuda yeterli bilgi bulamadım. Mulholland Drive'ın başka bir yönü hakkında soru sorabilir misiniz? Örneğin karakterler, semboller veya film sahneleri hakkında sorular sorabilirsiniz."


def build_system_prompt(context_docs: List[Dict[str, Any]]) -> str:
    """Sistem prompt'unu oluştur"""
    if not context_docs:
        return """Mulholland Drive filmi hakkında genel bilgi ver. Eğer spesifik bir konu hakkında yeterli bilgi yoksa, 
        kullanıcıya film hakkında sorabileceği diğer konuları öner. Örneğin karakterler, semboller, sahneler veya temalar hakkında sorular sorabilir."""
    
    # Kaynak metinleri özetle ve birleştir
    context_summary = []
    for d in context_docs:
        text = d.get("text", "")
        # Metni kısalt ve önemli kısımları al
        if len(text) > 600:
            text = text[:600] + " …"
        context_summary.append(text)
    
    # Tüm kaynakları tek bir metin olarak birleştir
    combined_context = " ".join(context_summary)
    
    system = (
        "Sen Mulholland Drive filmi uzmanısın. "
        "Aşağıdaki bilgileri kullanarak sorulan soruyu doğrudan ve özlü şekilde yanıtla.\n\n"
        f"Film bilgileri: {combined_context}\n\n"
        "Sadece sorulan soruya odaklan, gereksiz detaylara girme."
    )
    
    return system


def main():
    """Ana Streamlit uygulaması"""
    st.set_page_config(
        page_title="Mulholland Drive Analiz Asistanı",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Mulholland Drive Temalı CSS - Göz Yormayan ve Kullanıcı Dostu
    st.markdown("""
    <style>
    /* Ana arka plan - Color Hunt paleti ile */
    .main {
        background: linear-gradient(135deg, #2b3467 0%, #bad7e9 50%, #2b3467 100%);
        color: #fcffe7;
    }
    
    .stApp {
        background: linear-gradient(135deg, #2b3467 0%, #bad7e9 50%, #2b3467 100%);
    }
    
    /* Başlık stili - Color Hunt paleti ile */
    .film-title {
        color: #eb455f;
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(43, 52, 103, 0.5);
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    
    .film-subtitle {
        color: #bad7e9;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-style: italic;
        opacity: 0.9;
    }
    
    /* Hoş geldiniz kutusu - Color Hunt paleti ile */
    .welcome-box {
        background: linear-gradient(145deg, rgba(43, 52, 103, 0.9), rgba(186, 215, 233, 0.2));
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(235, 69, 95, 0.3);
        box-shadow: 0 8px 32px rgba(43, 52, 103, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Soru kutusu - Color Hunt paleti ile */
    .question-box {
        background: linear-gradient(145deg, rgba(43, 52, 103, 0.8), rgba(186, 215, 233, 0.1));
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid rgba(186, 215, 233, 0.4);
        box-shadow: 0 6px 24px rgba(43, 52, 103, 0.2);
        backdrop-filter: blur(8px);
    }
    
    /* Input alanları - Color Hunt paleti ile */
    .stTextInput > div > div > input {
        background: rgba(252, 255, 231, 0.95);
        border: 2px solid #bad7e9;
        color: #2b3467;
        border-radius: 10px;
        font-size: 15px;
        padding: 12px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #eb455f;
        box-shadow: 0 0 15px rgba(235, 69, 95, 0.3);
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(252, 255, 231, 0.95);
        border: 2px solid #bad7e9;
        color: #2b3467;
        border-radius: 10px;
        font-size: 15px;
        padding: 15px;
        transition: all 0.3s ease;
        min-height: 120px;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #eb455f;
        box-shadow: 0 0 15px rgba(235, 69, 95, 0.3);
    }
    
    /* Butonlar - Color Hunt paleti ile */
    .stButton > button {
        background: linear-gradient(45deg, #eb455f, #2b3467);
        color: #fcffe7;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(235, 69, 95, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(235, 69, 95, 0.4);
        background: linear-gradient(45deg, #2b3467, #eb455f);
    }
    
    /* Örnek sorular - Temiz tasarım */
    .example-questions {
        background: linear-gradient(145deg, rgba(186, 215, 233, 0.2), rgba(252, 255, 231, 0.1));
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #eb455f;
        border-right: 4px solid #eb455f;
    }
    
    /* Expander stili */
    .stExpander {
        background: rgba(43, 52, 103, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(235, 69, 95, 0.3);
    }
    
    .stExpander > div > div {
        background: rgba(43, 52, 103, 0.05);
        border-radius: 10px;
    }
    
    /* Expander başlığı */
    .stExpander > div > div > div {
        color: #eb455f;
        font-weight: 600;
        font-size: 16px;
    }
    
    /* Yazı boyutları - Color Hunt paleti ile */
    h1 {
        font-size: 2rem !important;
        color: #eb455f !important;
        text-shadow: 1px 1px 2px rgba(43, 52, 103, 0.5) !important;
    }
    
    h2 {
        font-size: 1.6rem !important;
        color: #bad7e9 !important;
    }
    
    h3 {
        font-size: 1.3rem !important;
        color: #eb455f !important;
    }
    
    p {
        font-size: 15px !important;
        color: #fcffe7 !important;
        line-height: 1.6 !important;
    }
    
    .stMarkdown {
        font-size: 15px !important;
        color: #fcffe7 !important;
        line-height: 1.6 !important;
    }
    
    /* Running yazısını gizle */
    .stSpinner > div {
        display: none;
    }
    
    /* Placeholder metni - Color Hunt paleti ile */
    .stTextArea textarea::placeholder {
        color: #bad7e9;
        font-style: italic;
    }
    
    /* Footer - Color Hunt paleti ile */
    .footer {
        background: rgba(43, 52, 103, 0.8);
        border-radius: 10px;
        padding: 20px;
        margin-top: 30px;
        text-align: center;
        border-top: 2px solid rgba(235, 69, 95, 0.3);
    }
    
    /* Scroll bar özelleştirmesi - Color Hunt paleti ile */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #2b3467;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #eb455f;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #bad7e9;
    }
    
    /* Streamlit menülerini tamamen gizle */
    .stDeployButton {
        display: none !important;
    }
    
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    div[data-testid="stHeader"] {
        display: none !important;
    }
    
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    
    div[data-testid="stDeployButton"] {
        display: none !important;
    }
    
    /* Hamburger menü ve deploy butonu için ek kurallar */
    .stApp > header {
        display: none !important;
    }
    
    .stApp > div[data-testid="stHeader"] {
        display: none !important;
    }
    
    .stApp > div[data-testid="stToolbar"] {
        display: none !important;
    }
    
    .stApp > div[data-testid="stDeployButton"] {
        display: none !important;
    }
    
    /* Tüm header elementlerini gizle */
    header {
        display: none !important;
    }
    
    /* Deploy butonu için spesifik kurallar */
    button[title="Deploy"] {
        display: none !important;
    }
    
    /* Cevap kutusu - Film tarzında (animasyonsuz) */
    .answer-box {
        background: linear-gradient(145deg, rgba(252, 255, 231, 0.95), rgba(186, 215, 233, 0.1));
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border: 2px solid #eb455f;
        box-shadow: 0 8px 32px rgba(43, 52, 103, 0.3);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .answer-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #eb455f, #bad7e9, #eb455f);
    }
    
    .answer-text {
        color: #2b3467;
        font-size: 16px;
        line-height: 1.8;
        font-weight: 500;
        text-align: justify;
        font-family: 'Georgia', 'Times New Roman', serif;
        text-shadow: 0.5px 0.5px 1px rgba(43, 52, 103, 0.1);
    }
    
    .answer-text::first-letter {
        font-size: 2.5em;
        float: left;
        line-height: 0.8;
        margin-right: 8px;
        margin-top: 4px;
        color: #eb455f;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(43, 52, 103, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Ana başlık - Film noir tarzında
    st.markdown('<h1 class="film-title">🎬 MULHOLLAND DRIVE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="film-subtitle">Film hakkında sorularınızı sorun, derinlemesine analiz alın</p>', unsafe_allow_html=True)
    
    # Hoş geldin mesajı - Color Hunt paleti ile
    st.markdown("""
    <div class="welcome-box">
        <h4 style="color: #eb455f; text-align: center; text-shadow: 1px 1px 2px rgba(43, 52, 103, 0.5);">🎭 HOLLYWOOD'UN GİZEMLİ YOLLARI</h4>
        <p style="text-align: center; margin: 0; color: #bad7e9; font-style: italic; line-height: 1.6;">
            Mulholland Drive'ın karanlık labirentlerinde kaybolmuş hikayeleri keşfedin. 
            Her soru, filmin derinliklerinde saklı bir sırrı ortaya çıkarır.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar ayarları (gizli)
    chunks_path = "chunks.jsonl"
    top_k = 5
    
    # API anahtarı kontrolü (arka planda)
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Chunks yükleme (arka planda)
    if "doc_embeddings" not in st.session_state:
        if os.path.exists(chunks_path):
            try:
                emb, chunks = load_embeddings_and_chunks(chunks_path)
                st.session_state["doc_embeddings"] = emb
                st.session_state["chunks"] = chunks
            except Exception as e:
                st.error("Sistem hazırlanırken bir hata oluştu. Lütfen sayfayı yenileyin.")
                st.stop()
        else:
            st.error("Film veritabanı bulunamadı. Lütfen sistem yöneticisiyle iletişime geçin.")
            st.stop()
    
    # Ana sohbet arayüzü
    
    # Örnek sorular - Expander ile
    with st.expander("💡 Örnek Sorular", expanded=False):
        examples = [
            "Mulholland Drive'ın gerçek hikayesi nedir?",
            "Betty ve Diane aynı kişi mi?",
            "Club Silencio'nun gizemi ne?",
            "Mavi anahtar neyi temsil ediyor?",
            "Filmdeki rüya ve gerçeklik ayrımı nasıl?",
            "Mulholland Drive'ın ana teması nedir?"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"💫 {example}", key=f"example_{example}", use_container_width=True):
                    st.session_state["user_question"] = example
    
    # Soru girişi
    user_question = st.text_area(
        "Sorunuzu yazın:",
        value=st.session_state.get("user_question", ""),
        height=100,
        placeholder="Örn: Mulholland Drive'ın konusu nedir?"
    )
    
    # Sor butonu
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ask_button = st.button("🔍 Sor", type="primary", use_container_width=True)
    
    # Yanıt üretme
    if ask_button:
        if not user_question.strip():
            st.warning("Lütfen bir soru girin.")
        elif not api_key:
            st.error("Sistem şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.")
        else:
            with st.spinner("🤔 Sorunuz analiz ediliyor, lütfen bekleyin..."):
                try:
                    # Progress mesajları
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Embedding oluştur
                    status_text.text("📝 Sorunuz işleniyor...")
                    progress_bar.progress(20)
                    query_vec = embed_query(user_question)
                    if query_vec is None:
                        st.error("Soru işlenirken bir hata oluştu.")
                        return
                    
                    # En benzer parçaları bul
                    status_text.text("🔍 İlgili bilgiler aranıyor...")
                    progress_bar.progress(50)
                    results = retrieve_top_k(
                        query_vec, 
                        st.session_state["doc_embeddings"], 
                        st.session_state["chunks"], 
                        top_k=top_k
                    )
                    
                    if not results:
                        st.warning("Bu konuda yeterli bilgi bulunamadı.")
                        return
                    
                    # Sistem prompt'u oluştur
                    status_text.text("🧠 Analiz hazırlanıyor...")
                    progress_bar.progress(80)
                    system_prompt = build_system_prompt(results)
                    
                    # Gemini'den yanıt al
                    status_text.text("✨ Yanıt oluşturuluyor...")
                    progress_bar.progress(100)
                    answer = call_gemini(system_prompt, user_question)
                    
                    # Progress'i temizle
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Sonuçları göster - Film tarzında
                    st.markdown(f"""
                    <div class="answer-box">
                        <div class="answer-text">{answer}</div>
                    </div>
                    """, unsafe_allow_html=True)
                            
                except Exception as e:
                    st.markdown(f"""
                    <div class="answer-box">
                        <div class="answer-text">
                            Üzgünüm, bu konuda yeterli bilgi bulamadım. Mulholland Drive'ın başka bir yönü hakkında soru sorabilir misiniz? 
                            <br><br>
                            <strong>Öneriler:</strong><br>
                            • Karakterler hakkında sorular (Betty, Diane, Rita)<br>
                            • Semboller hakkında sorular (mavi anahtar, Club Silencio)<br>
                            • Film sahneleri hakkında sorular<br>
                            • Temalar hakkında sorular (rüya, gerçeklik, Hollywood)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Footer - Color Hunt paleti ile
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p style="color: #bad7e9; font-size: 0.9rem; font-style: italic; margin: 0;">
            "No hay banda. It is all an illusion." - Mulholland Drive
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()