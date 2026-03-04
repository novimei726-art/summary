import torch
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import types
from typing import List
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase
from transformers import MarianMTModel, MarianTokenizer


# -------------------------------------------------------
# STYLE UI (opsional)
# -------------------------------------------------------
try:
    from style_ui import load_custom_style
    load_custom_style()
except Exception as e:
    st.warning(f"Style gagal dimuat: {e}")



st.set_page_config(page_title="Skripsi Text Summarization", layout="centered")
st.title("Teks Summarization – IndoBART")
st.caption(
    "Inferensi disamakan 1:1 dengan Colab: LoRA/PEFT, Tokenizer IndoNLG, "
    "dan parameter decoding yang sama."
)
st.caption(
    "Berita: Mediacenter.com, MMCKalteng.co.ic, kalteng.go.id, detik.com."
)

# -------------------------------------------------------
# KONFIGURASI GLOBAL
# -------------------------------------------------------
MAX_MODEL_INPUT = 800
CHUNK_STRIDE = 400
MAX_SUMMARY_LENGTH = 100
ERROR_TEXT = "Ringkasan gagal dibuat karena error model."

# -------------------------------------------------------
# MODEL TRANSLATE LOKAL (NLLB 200)
# -------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model terjemahan Indonesia → Inggris (NLLB)...")
def load_mt_model():
    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return tokenizer, model, device

mt_tokenizer, mt_model, mt_device = load_mt_model()


def normalize_id_text_for_translation(text: str) -> str:
    """
    Normalisasi teks Indonesia sebelum dikirim ke model terjemahan:
    - rapikan spasi
    - kapitalisasi awal kalimat
    (bersifat umum, tidak tergantung isi berita tertentu)
    """
    if not text:
        return text

    # Paksa ada spasi setelah tanda titik
    text = re.sub(r"([.!?])(\S)", r"\1 \2", text)

    # Rapikan whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Kapitalisasi awal kalimat
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]

    return " ".join(sentences)

def postprocess_summary(text: str) -> str:
    """
    Membersihkan ringkasan (versi umum, bisa untuk semua berita):
    - merapikan spasi
    - menghapus 'pj.' di awal kalimat
    - menghapus kata berulang seperti 'strategi strategis' -> 'strategi'
    - kapitalisasi awal kalimat
    - memastikan diakhiri tanda titik
    """
    if not text:
        return text

    # Rapikan spasi dulu
    text = re.sub(r"\s+", " ", text).strip()

    # Hapus 'pj.' di awal
    text = re.sub(r"^\s*pj\.\s*", "", text, flags=re.IGNORECASE)

    # Hilangkan kata yang dobel berurutan
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

    # Pecah jadi kalimat dan kapitalisasi awal tiap kalimat
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > 1 and s[0].isalpha() and not s[0].isupper():
            s = s[0].upper() + s[1:]
        cleaned.append(s)

    result = " ".join(cleaned).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result

# === Domain Lexicon untuk berita pemerintahan ===
DOMAIN_TERMS = {
    # Instansi
    "dinas kesehatan": "Health Office",
    "dinas komunikasi dan informatika": "Department of Communication and Informatics",
    "diskominfo": "Diskominfo",
    "dinas pendidikan": "Education Office",

    # Administrasi
    "tata kelola": "governance",
    "persandian": "cryptography",
    "kabupaten": "Regency",
    "kota": "City",

    # Wilayah
    "palangka raya": "Palangka Raya",
    "kalteng": "Central Kalimantan",
}

def postprocess_translation(en_text: str, src_text: str = "") -> str:
    """
    Post-processing umum untuk hasil terjemahan:
    - merapikan spasi
    - menghapus kata/phrase berulang
    - merapikan spasi sebelum tanda baca
    - memperbaiki beberapa pola bahasa Inggris yang jelek namun cukup umum
    - kapitalisasi awal kalimat

    Bersifat umum, tidak di-hardcode untuk 1 berita atau 1 nama.
    """
    if not en_text:
        return en_text

    out = en_text

    for k, v in DOMAIN_TERMS.items():
        out = re.sub(
            rf"\b{k}\b",
            v,
            out,
            flags=re.IGNORECASE
        )

    # 1) Rapikan spasi berlebih
    out = re.sub(r"\s+", " ", out).strip()

    # 2) Perbaiki pola umum yang sering muncul di banyak terjemahan
    #    (tidak terkait nama orang / tempat tertentu)
    replacements = {
        r"\bthe the\b": "the",
        r"\ba a\b": "a",
        r"\ban an\b": "an",
        r"\bof of\b": "of",

        r"break[- ]up ceremony": "farewell and welcoming ceremony",
        r"breakup of the head prosecutor of": "farewell and welcoming ceremony for the Chief Prosecutor of",
        r"break[- ]up of the head prosecutor of": "farewell and welcoming ceremony for the Chief Prosecutor of",

        # === QUALITY FIXES (GLOBAL) ===
        r"\bnot habitable\b": "substandard housing",
        r"\buninhabitable housing\b": "substandard housing",
        r"\bagainst the national program\b": "in support of the national program",
        r"\bof the highway\b": "",
        r"\bhousing and housing services\b": "housing and settlement services",
        r"\bMEDIA CENTR\b": "MEDIA CENTER",
    }

    for pattern, repl in replacements.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    # 3) Hilangkan kata yang sama persis berurutan (umum)
    #    contoh: "important important" -> "important"
    out = re.sub(r"\b(\w+)\s+\1\b", r"\1", out, flags=re.IGNORECASE)

    # 4) Rapikan spasi sebelum tanda baca
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)

    # 5) Kapitalisasi awal kalimat
    sentences = re.split(r"(?<=[.!?])\s+", out)
    fixed = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if s[0].islower():
            s = s[0].upper() + s[1:]
        fixed.append(s)

    out = " ".join(fixed).strip()

    return out

def translate_to_english(text: str) -> str:
    if not text or text == ERROR_TEXT:
        return ""

    try:
        # 1️⃣ Normalisasi + hapus entity token IndoBART
        cleaned_id_text = normalize_id_text_for_translation(text)
        cleaned_id_text = re.sub(
            r'__ENT\d+(_[A-Z]+)?_+',
            '',
            cleaned_id_text
        )
        cleaned_id_text = re.sub(r'\s+', ' ', cleaned_id_text).strip()

        # 2️⃣ Tokenisasi NLLB (WAJIB set language)
        mt_tokenizer.src_lang = "ind_Latn"

        encoded = mt_tokenizer(
            cleaned_id_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(mt_device)

        # 3️⃣ Generate terjemahan
        with torch.no_grad():
            generated_tokens = mt_model.generate(
                **encoded,
                forced_bos_token_id=mt_tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_new_tokens=300,
                num_beams=5,
                no_repeat_ngram_size=3,
            )

        raw = mt_tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True
        )

        # 4️⃣ Postprocess ringan
        cleaned = postprocess_translation(raw)
        return cleaned

    except Exception as e:
        st.error(f"⚠️ Terjemahan gagal: {e}")
        return ""


# -------------------------------------------------------
# UTILITAS SCRAPING
# -------------------------------------------------------
def ekstrak_tanggal_manual(url, html):
    soup = BeautifulSoup(html, "html.parser")
    teks = soup.get_text(separator=" ")

    pola_ind = re.findall(
        r"\b(\d{1,2}\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4})\b",
        teks,
    )
    if pola_ind:
        return pola_ind[0][0]

    pola_iso = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", teks)
    if pola_iso:
        return pola_iso[0]

    pola_slash = re.findall(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", teks)
    if pola_slash:
        return pola_slash[0]

    return "Tanggal tidak tersedia"


def ambil_berita_dari_url(url):
    try:
        artikel = Article(url, language="id")
        artikel.download()
        artikel.parse()
        judul = artikel.title or "Judul tidak ditemukan"
        teks = artikel.text or ""

        if artikel.publish_date:
            try:
                tanggal_str = artikel.publish_date.strftime("%d %B %Y")
            except Exception:
                tanggal_str = str(artikel.publish_date)
        else:
            resp = requests.get(url, timeout=10)
            tanggal_str = ekstrak_tanggal_manual(url, resp.text)

        if not teks or len(teks) < 100:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            teks = " ".join(p.get_text() for p in soup.find_all("p"))
            if not teks:
                raise Exception("Teks artikel terlalu pendek atau tidak ditemukan.")

        return judul, tanggal_str, teks
    except Exception as e:
        st.error(f"⚠️ Gagal mengambil berita: {e}")
        return None, None, None


def bersih(teks: str) -> str:
    return re.sub(r"\s+", " ", teks).strip()


def lead_summary(text: str) -> str:
    """
    Lead-like extractive summary: kalimat pertama berita (setelah 'MEDIA CENTER,...' dll)
    """
    cleaned = re.sub(
        r"^MEDIA CENTER[,|\s].*?[–-]\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    kalimat = re.split(r"(?<=[.!?])\s+", cleaned)
    kalimat = [k.strip() for k in kalimat if len(k.strip()) > 10]
    return kalimat[0] if kalimat else ""


# -------------------------------------------------------
# PATCH TOKENIZER (Pad fix)
# -------------------------------------------------------
def _compat_pad(
    self,
    encoded_inputs,
    padding=False,
    max_length=None,
    pad_to_multiple_of=None,
    return_attention_mask=None,
    return_tensors=None,
    verbose=True,
    **kwargs,
):
    return PreTrainedTokenizerBase.pad(
        self,
        encoded_inputs,
        padding=padding,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=return_attention_mask,
        return_tensors=return_tensors,
        verbose=verbose,
    )

# -------------------------------------------------------
# LOAD MODEL & TOKENIZER
# -------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model IndoBART dan tokenizer...")
def load_model_and_tokenizer():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # ⚠️ GANTI INI dengan model Anda di HuggingFace Hub
    # Contoh: "username/model-name" atau gunakan model base "indobenchmark/indobart-v2"
    model_path = "indobenchmark/indobart-v2"  # Model base, bukan fine-tuned
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load model ---
    try:
        st.write("Memuat model:", model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        ).to(device)
        model.eval()
    except Exception as e:
        st.error(f"❌ Gagal memuat model. Detail: {e}")
        st.stop()

    # --- Load tokenizer ---
    try:
        st.write("Memuat tokenizer AutoTokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # patch pad() jika diperlukan
        def _compat_pad_local(self, encoded_inputs, padding=False, max_length=None,
                              pad_to_multiple_of=None, return_attention_mask=None,
                              return_tensors=None, verbose=True, **kwargs):
            return PreTrainedTokenizerBase.pad(
                self,
                encoded_inputs,
                padding=padding,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_tensors=return_tensors,
                verbose=verbose
            )

        tokenizer.pad = types.MethodType(_compat_pad_local, tokenizer)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    except Exception as e:
        st.error(f"❌ Gagal memuat tokenizer. Detail: {e}")
        st.stop()

    return model, tokenizer, device

# Init
model, tokenizer, device = load_model_and_tokenizer()

# -------------------------------------------------------
# CHUNKING & SUMMARIZATION (disamakan dengan Colab)
# -------------------------------------------------------
def chunk_text(text, tokenizer, max_input_length=MAX_MODEL_INPUT, stride=CHUNK_STRIDE):
    inputs = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
    input_ids = inputs["input_ids"][0]
    total_len = input_ids.size(0)
    chunks = []

    for i in range(0, total_len, stride):
        end = min(i + max_input_length, total_len)
        chunk_ids = input_ids[i:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == total_len:
            break

    return chunks


def summarize_chunk(
    text,
    model,
    tokenizer,
    device,
    max_input_length=MAX_MODEL_INPUT,
    max_output_length=MAX_SUMMARY_LENGTH,
):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_output_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,  # disamakan dengan Colab
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_long_text(
    text,
    model,
    tokenizer,
    device,
    max_input_length=MAX_MODEL_INPUT,
    stride=CHUNK_STRIDE,
    max_output_length=MAX_SUMMARY_LENGTH,
    num_sentences=3,
):
    chunks = chunk_text(text, tokenizer, max_input_length, stride)
    summaries = [
        summarize_chunk(
            c, model, tokenizer, device,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )
        for c in chunks
    ]

    combined = " ".join(summaries)
    combined = re.sub(r"\s+", " ", combined).strip()

    sentences = re.split(r"(?<=[.!?])\s+", combined)
    sentences = [s.strip() for s in sentences if s.strip()]

    selected = sentences[:num_sentences]
    final = " ".join(selected).strip()

    if final and final[-1] not in ".!?":
        final += "."
    return final


def ambil_satu_kalimat_dari_ringkasan(teks_ringkasan: str) -> str:
    """
    Ambil 1 kalimat yang enak dibaca dari ringkasan multi-kalimat.
    Jika kalimat pertama terlalu pendek (misal 'pj.'), digabung dengan kedua.
    """
    kalimat = re.split(r"(?<=[.!?])\s+", teks_ringkasan)
    kalimat = [k.strip() for k in kalimat if k.strip()]

    if not kalimat:
        return teks_ringkasan.strip()

    if len(kalimat[0]) <= 10 and len(kalimat) > 1:
        return (kalimat[0] + " " + kalimat[1]).strip()

    return kalimat[0]

# -------------------------------------------------------
# ANTI-HALLUCINATION SEDERHANA (UMUM)
# -------------------------------------------------------
RISK_TITLES = [
    "gubernur",
    "bupati",
    "wali kota",
    "walikota",
    "menteri",
    "presiden",
    "wakil presiden",
    "kapolda",
    "kapolres",
]

def anti_hallucination(summary: str, original_text: str) -> str:
    """
    Mengurangi halusinasi jabatan umum:
    - jika ringkasan menyebut jabatan tertentu (gubernur, bupati, dst)
      tetapi teks asli tidak menyebut jabatan itu sama sekali,
      maka kata jabatannya dihapus dari ringkasan.
    """
    s = summary
    s_low = s.lower()
    src_low = original_text.lower()

    for title in RISK_TITLES:
        if title in s_low and title not in src_low:
            pattern = rf"{title}\s+[a-zA-Z\. ]+,"
            s = re.sub(pattern, "", s, flags=re.IGNORECASE)
            s = re.sub(rf"\b{title}\b", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s+", " ", s).strip()
            s_low = s.lower()

    return s.strip()

# -------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------
for key in [
    "judul",
    "tanggal",

    # 🔹 tambahan untuk input manual
    "judul_manual",
    "tanggal_manual",
    "sumber_manual",

    "text_input",
    "ringkasan_1",
    "ringkasan_3",
    "full_text",
    "lead_summary",
    "url_input",
    "ringkasan_1_en",
    "ringkasan_3_en",
    "lead_summary_en",
]:
    if key not in st.session_state:
        st.session_state[key] = ""

# -------------------------------------------------------
# INPUT
# -------------------------------------------------------
st.markdown("## 📝 Pilih Metode Input")

mode = st.radio(
    "Pilih metode:",
    ["URL Berita", "Input Manual"],
    horizontal=True,
)

if mode == "URL Berita":
    st.markdown("### 🌐 Masukkan URL Berita")
    url = st.text_input(
        "Masukkan URL Berita:",
        placeholder="https://...",
        label_visibility="collapsed",
        key="url_input",
    )
    manual_text = ""

else:
    st.markdown("### ✍️ Isi data dibawah ini:")

    judul_manual = st.text_input("Judul Berita :")
    tanggal_manual = st.date_input("Tanggal Berita :")
    sumber_manual = st.text_input("Link Sumber Berita :")

    manual_text = st.text_area(
        "Tempelkan teks berita di sini:",
        height=250,
        placeholder="Salin isi berita di sini...",
    )
    url = ""

# -------------------------------------------------------
# TOMBOL PROSES
# -------------------------------------------------------
if st.button("📰 Buat Ringkasan", key="btn_ringkas"):

    if mode == "URL Berita" and not url.strip():
        st.warning("Masukkan URL terlebih dahulu.")
    elif mode == "Input Manual" and not manual_text.strip():
        st.warning("Tempelkan teks terlebih dahulu.")
    else:
        st.session_state["ringkasan_1_en"] = ""
        st.session_state["ringkasan_3_en"] = ""
        st.session_state["lead_summary_en"] = ""
        st.session_state["full_text"] = ""

        with st.spinner("Sedang memproses dan merangkum... ⏳"):
            st.session_state["source_mode"] = mode

            if mode == "URL Berita":
                judul_s, tanggal_s, teks = ambil_berita_dari_url(url)
                sumber_s = url
            else:
                judul_s = judul_manual or "Input Manual"
                tanggal_s = str(tanggal_manual)
                sumber_s = sumber_manual or "Input Manual"
                teks = manual_text

            if teks and len(teks.strip()) > 50:
                teks_bersih = bersih(teks)

                st.session_state["judul"] = judul_s
                st.session_state["tanggal"] = tanggal_s
                st.session_state["sumber_manual"] = sumber_s
                st.session_state["full_text"] = teks_bersih
                st.session_state["text_input"] = teks_bersih

                # --- Ringkasan maks. 3 kalimat dari model ---
                try:
                    abstr_3_raw = summarize_long_text(
                        teks_bersih,
                        model,
                        tokenizer,
                        device=device,
                        max_input_length=MAX_MODEL_INPUT,
                        stride=CHUNK_STRIDE,
                        max_output_length=MAX_SUMMARY_LENGTH,
                        num_sentences=3,
                    )
                except Exception as e:
                    st.error(f"Ringkasan abstraktif (3 kalimat) error: {e}")
                    abstr_3_raw = ERROR_TEXT

                if abstr_3_raw != ERROR_TEXT:
                    abstr_3_clean = postprocess_summary(abstr_3_raw)
                    abstr_3_clean = anti_hallucination(abstr_3_clean, teks_bersih)

                    abstr_1_raw = ambil_satu_kalimat_dari_ringkasan(abstr_3_clean)
                    abstr_1_clean = postprocess_summary(abstr_1_raw)
                    abstr_1_clean = anti_hallucination(abstr_1_clean, teks_bersih)

                    st.session_state["ringkasan_3"] = abstr_3_clean
                    st.session_state["ringkasan_1"] = abstr_1_clean
                else:
                    st.session_state["ringkasan_3"] = abstr_3_raw
                    st.session_state["ringkasan_1"] = ERROR_TEXT

                st.session_state["lead_summary"] = lead_summary(teks_bersih)

# -------------------------------------------------------
# OUTPUT
# -------------------------------------------------------
# Ringkasan Abstraktif 1 Kalimat
if st.session_state.get("ringkasan_1"):
    st.markdown(
        f"""
        <div class="summary-box summary-abstractive">
            <b>Ringkasan Abstraktif (1 Kalimat):</b>
            <p>{st.session_state['ringkasan_1']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Terjemahan Inggris – 1 Kalimat
    if st.session_state.get("ringkasan_1_en"):
        st.markdown(
            f"""
            <div class="summary-box translation-en">
                <b>English Translation (1 Sentence):</b>
                <p>{st.session_state['ringkasan_1_en']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Ringkasan Abstraktif Maks. 3 Kalimat
if st.session_state.get("ringkasan_3"):
    st.markdown(
        f"""
        <div class="summary-box summary-abstractive">
            <b>Ringkasan Abstraktif (3 Kalimat):</b>
            <p>{st.session_state['ringkasan_3']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Terjemahan Inggris – 3 Kalimat
    if st.session_state.get("ringkasan_3_en"):
        st.markdown(
            f"""
            <div class="summary-box translation-en">
                <b>English Translation (3 Sentences):</b>
                <p>{st.session_state['ringkasan_3_en']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Ringkasan Ekstraktif (Lead-like)
if st.session_state.get("lead_summary"):
    st.markdown(
        f"""
        <div class="summary-box summary-extractive">
            <b>Ringkasan Ekstraktif (Lead-like):</b>
            <p>{st.session_state['lead_summary']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Terjemahan Inggris – Lead-like
    if st.session_state.get("lead_summary_en"):
        st.markdown(
            f"""
            <div class="summary-box translation-en">
                <b>English Translation (Lead-like):</b>
                <p>{st.session_state['lead_summary_en']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------------------------------------
# TOMBOL TERJEMAHKAN SEMUA RINGKASAN (GLOBAL)
# -------------------------------------------------------
if (
    st.session_state.get("ringkasan_1")
    or st.session_state.get("ringkasan_3")
    or st.session_state.get("lead_summary")
):
    st.markdown("---")
    if st.button("🇬🇧 Terjemahkan Semua Ringkasan ke Bahasa Inggris"):
        with st.spinner("Menerjemahkan semua ringkasan... ⏳"):
            if st.session_state.get("ringkasan_1"):
                st.session_state["ringkasan_1_en"] = translate_to_english(
                    st.session_state["ringkasan_1"]
                )

            if st.session_state.get("ringkasan_3"):
                st.session_state["ringkasan_3_en"] = translate_to_english(
                    st.session_state["ringkasan_3"]
                )

            if st.session_state.get("lead_summary"):
                st.session_state["lead_summary_en"] = translate_to_english(
                    st.session_state["lead_summary"]
                )

# Detail berita hanya kalau ringkasan tidak error
if (
    st.session_state.get("ringkasan_1")
    and st.session_state["ringkasan_1"] != ERROR_TEXT
):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align:center; margin-top:30px;'>📋 Detail Isi Berita</h3>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="detail-box"><b>Judul:</b><br>{st.session_state["judul"]}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="detail-box"><b>Tanggal:</b><br>{st.session_state["tanggal"]}</div>',
            unsafe_allow_html=True,
        )

    with st.columns(1)[0]:
        if st.session_state.get("source_mode") == "URL Berita":
            st.markdown(
                f'<div class="detail-box"><b>URL:</b><br>'
                f'<a href="{st.session_state["url_input"]}" target="_blank">'
                f'{st.session_state["url_input"]}</a></div>',
                unsafe_allow_html=True,
            )
        else:
            src = st.session_state.get("sumber_manual", "Input Manual")

            if src.startswith("http"):
                st.markdown(
                    f'<div class="detail-box"><b>Sumber:</b><br>'
                    f'<a href="{src}" target="_blank">{src}</a></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="detail-box"><b>Sumber:</b><br>{src}</div>',
                    unsafe_allow_html=True,
                )


    with st.expander("📖 Lihat Isi Berita Lengkap"):
        st.markdown(
            f'<div class="detail-box text-full">{st.session_state["full_text"]}</div>',
            unsafe_allow_html=True,
        )
