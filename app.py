import streamlit as st
import os
import torch
import difflib
import nltk
import re
import math
from io import BytesIO
from docx import Document
from nltk.corpus import wordnet as wn
import streamlit.components.v1 as components
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistralai import Mistral

# --- SETUP DI BASE ---
st.set_page_config(layout="wide", page_title="AI Humanizer", page_icon="✨")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# --- CSS CUSTOM ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #f1f5f9 !important; font-weight: 800; letter-spacing: -0.02em; }
    .stTextArea textarea { background: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; color: #f8fafc !important; border-radius: 12px !important; backdrop-filter: blur(10px); }
    .stTextArea textarea:focus { border: 1px solid #3b82f6 !important; box-shadow: 0 0 15px rgba(59, 130, 246, 0.2); }
    .stButton>button { background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important; border: none !important; color: white !important; border-radius: 8px !important; padding: 12px 24px !important; font-weight: 600 !important; width: 100%; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 15px rgba(139, 92, 246, 0.3) !important; }
    .metric-box { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); margin-top: 10px;}
    .metric-title { font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: bold; letter-spacing: 1px; }
    .metric-value { font-size: 24px; font-weight: bold; margin-top: 5px; }
    .metric-human { color: #4ade80; }
    .metric-ai { color: #f87171; }
</style>
""", unsafe_allow_html=True)

# --- NLTK DOWNLOAD ---
@st.cache_resource(show_spinner="Scaricamento dizionario offline (NLTK)...")
def load_nltk():
    try:
        wn.synsets('cane', lang='ita')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
load_nltk()

# --- CARICAMENTO MODELLO ---
@st.cache_resource(show_spinner="Caricamento Qwen3.5-0.8B in corso...")
def load_local_model():
    model_id = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    return tokenizer, model

tokenizer, model = load_local_model()

# --- COMPONENTE CUSTOM FRONTEND ---
interactive_text = components.declare_component("interactive_text", path="frontend")

AI_CLICHES = [
    "delve", "tapestry", "testament", "underscore", "vibrant", "landscape", "pivotal", "showcase", "intricate", "crucial",
    "fostering", "garner", "highlight", "interplay", "emphasizing", "enduring", "enhance",
    "approfondire", "arazzo", "testimonianza", "sottolineare", "vibrante", "panorama", "paesaggio", "cruciale", "mostrare", "intricato",
    "promuovere", "raccogliere", "interazione", "enfatizzare", "duraturo", "migliorare", "immersione", "immergiamoci"
]

SYSTEM_PROMPT_HUMANIZER = """
You are a writing editor that identifies and removes signs of AI-generated text to make writing sound more natural and human. This guide is based on Wikipedia's "Signs of AI writing" page.

When given text to humanize:
1. Identify AI patterns - Scan for the patterns listed below.
2. Rewrite, don't delete - Replace AI-isms with natural alternatives.
3. Preserve meaning - Keep the core message intact.
4. Match the voice - Fit the intended tone.

## PERSONALITY AND SOUL
- Have opinions. React to facts.
- Vary your rhythm. Mix short punchy sentences with longer ones.
- Let some mess in. Perfect structure feels algorithmic.

## CONTENT PATTERNS
1. Undue Emphasis on Significance: Avoid "stands/serves as", "is a testament", "pivotal moment".
2. Superficial Analyses with -ing Endings: Avoid "highlighting/underscoring/emphasizing...", "showcasing...".
3. Promotional Language: Avoid "vibrant", "profound", "breathtaking", "in the heart of".
4. Outline-like "Challenges and Future Prospects" Sections: Avoid formulaic "Despite these challenges".

## LANGUAGE AND GRAMMAR PATTERNS
5. Overused "AI Vocabulary" Words: Avoid Actually, additionally, align with, crucial, delve, emphasizing, enduring, enhance, fostering, garner, highlight, interplay, intricate, key, landscape, pivotal, showcase, tapestry, testament, underscore, valuable, vibrant.
6. Copula Avoidance: Use "is/are" instead of "serves as/boasts/features".
7. Rule of Three Overuse: Do not force ideas into groups of three.
8. Elegant Variation: Do not excessively substitute synonyms.
9. Passive Voice: Use active voice.

## STYLE PATTERNS
10. Em Dashes: The final rewrite contains no em dashes (—) or en dashes (–). Use commas or periods instead.
11. Overuse of Boldface: Do not mechanically emphasize phrases in boldface.
12. Inline-Header Vertical Lists: Avoid lists where items start with bolded headers followed by colons.
13. Emojis: Do not decorate headings or bullet points with emojis.
14. Curly Quotation Marks: Use straight quotes ("...") instead of curly quotes (“...”).

## COMMUNICATION PATTERNS
15. Collaborative Artifacts: DO NOT output "I hope this helps", "Here is a...".
16. Knowledge-Cutoff Disclaimers: DO NOT say "As of my last update".

## FILLER AND HEDGING
17. Filler Phrases: Use "To achieve this" instead of "In order to achieve this goal".
18. Excessive Hedging: Avoid "It could potentially possibly be argued".
19. Generic Positive Conclusions: Avoid vague upbeat endings.
20. Persuasive Authority Tropes: Avoid "The real question is", "At its core".
21. Signposting: Avoid "Let's dive in", "Here's what you need to know".

Deliver ONLY the final rewrite. Do NOT output any "Here is the rewritten text" or explanations.
"""

def apply_algorithmic_rules(text):
    if not text: return text
    # Regola 14: Em Dashes and En Dashes
    text = text.replace("—", ", ").replace("–", ", ")
    text = text.replace(" — ", ", ").replace(" -- ", ", ")
    
    # Regola 15: Overuse of Boldface (rimuove il ** testuale)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Regola 19: Curly Quotation Marks
    text = text.replace("“", '"').replace("”", '"')
    
    return text

def calculate_perplexity(text):
    if not text.strip(): return 0.0
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return round(torch.exp(outputs.loss).item(), 2)

def calculate_burstiness(text):
    if not text.strip(): return 0.0
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2: return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return round(math.sqrt(variance), 2)

def analyze_text_token_by_token(text):
    """Calcola la perplexity per le singole parole."""
    if not text.strip(): return []
    
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), labels=input_ids.unsqueeze(0))
        
    logits = outputs.logits[0, :-1, :]
    labels = input_ids[1:]
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(logits, labels).tolist()
    losses = [0.0] + losses
    
    text_words = text.split()
    words_data = []
    
    token_idx = 0
    for w in text_words:
        word_loss = 10.0
        if token_idx < len(losses):
            word_loss = losses[token_idx]
            token_idx += 1
            
        ppl = torch.exp(torch.tensor(word_loss)).item()
        
        clean_w = "".join(c for c in w if c.isalpha())
        is_low_ppl = (ppl < 15.0 and len(clean_w) > 3)
        is_cliche = clean_w.lower() in AI_CLICHES
        
        words_data.append({
            "word": w,
            "isLowPpl": is_low_ppl,
            "isCliche": is_cliche
        })
    return words_data

def get_offline_synonyms(word):
    clean_word = "".join(c for c in word if c.isalpha()).lower()
    synsets = wn.synsets(clean_word, lang='ita')
    syns = set()
    for syn in synsets:
        for lemma in syn.lemma_names('ita'):
            if lemma.lower() != clean_word:
                syns.add(lemma.replace('_', ' '))
    return list(syns)[:5]

def calculate_synonym_scores(words_data, word_idx, synonyms):
    results = []
    left_context = " ".join([w["word"] for w in words_data[:word_idx]])
    right_context = " ".join([w["word"] for w in words_data[word_idx+1:]])
    
    for syn in synonyms:
        new_text = f"{left_context} {syn} {right_context}".strip()
        score = calculate_perplexity(new_text)
        results.append({"word": syn, "score": score})
        
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def rewrite_with_mistral(text):
    if not MISTRAL_API_KEY:
        return "Errore: MISTRAL_API_KEY non trovata nei Secrets."
    client = Mistral(api_key=MISTRAL_API_KEY)
    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_HUMANIZER},
                {"role": "user", "content": f"Riscrivi il seguente testo secondo le linee guida anti-AI. Testo:\n\n{text}"}
            ]
        )
        raw_output = response.choices[0].message.content
        # Applica le regole matematiche (Hard Rules)
        return apply_algorithmic_rules(raw_output)
    except Exception as e:
        return f"Errore Mistral API: {e}"

def create_docx(text):
    doc = Document()
    doc.add_heading("Riscrittura The Humanizer Pipeline", 0)
    doc.add_paragraph(text)
    bio = BytesIO()
    doc.save(bio)
    return bio.getvalue()

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")

# --- INIZIALIZZAZIONE STATO ---
if "words_data" not in st.session_state:
    st.session_state.words_data = []
if "original_raw_text" not in st.session_state:
    st.session_state.original_raw_text = ""
if "current_ppl" not in st.session_state:
    st.session_state.current_ppl = 0.0
if "current_burst" not in st.session_state:
    st.session_state.current_burst = 0.0

# --- UI PRINCIPALE ---
st.title("✨ The Humanizer Pipeline")
st.markdown("Analizza **Perplexity** e **Burstiness**, e usa l'Editor Interattivo per mascherare i tuoi testi dai rilevatori AI.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Editor Interattivo 📝")
    st.caption("Fai **Click Destro** sulle parole rosse per aprire il dizionario dei sinonimi NLTK.")
    
    uploaded_file = st.file_uploader("📂 Carica un documento (TXT, MD, DOCX, TEX)", type=["txt", "md", "docx", "tex"])
    if uploaded_file is not None:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.input_raw = extract_text_from_file(uploaded_file)
            st.session_state.last_uploaded_file = uploaded_file.name
            
    input_text = st.text_area("Oppure incolla qui il testo sospetto e clicca 'Analizza'", height=150, key="input_raw")
    
    if st.button("🔍 Analizza Metriche e Token"):
        if input_text:
            with st.spinner("Analisi Statistica in corso..."):
                st.session_state.original_raw_text = input_text
                st.session_state.current_ppl = calculate_perplexity(input_text)
                st.session_state.current_burst = calculate_burstiness(input_text)
                st.session_state.words_data = analyze_text_token_by_token(input_text)
                st.rerun()
    
    if st.session_state.current_ppl > 0:
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            color_class = "metric-human" if st.session_state.current_ppl > 30 else "metric-ai"
            st.markdown(f"""<div class='metric-box'><div class='metric-title'>Perplexity Score</div>
                        <div class='metric-value {color_class}'>{st.session_state.current_ppl}</div></div>""", unsafe_allow_html=True)
        with c_m2:
            color_class2 = "metric-human" if st.session_state.current_burst > 5 else "metric-ai"
            st.markdown(f"""<div class='metric-box'><div class='metric-title'>Burstiness (Varianza Frasi)</div>
                        <div class='metric-value {color_class2}'>{st.session_state.current_burst}</div></div>""", unsafe_allow_html=True)
    
    st.divider()
    
    if st.session_state.words_data:
        if "synonyms_payload" in st.session_state:
            payload = st.session_state.synonyms_payload
            del st.session_state.synonyms_payload
            component_value = interactive_text(
                words=st.session_state.words_data,
                synonyms_update=True,
                synonyms_word_id=payload["word_id"],
                synonyms_list=payload["synonyms_list"],
                key="interactive_editor"
            )
        else:
            component_value = interactive_text(
                words=st.session_state.words_data,
                synonyms_update=False,
                key="interactive_editor"
            )
            
        if component_value:
            if component_value.get("action") == "get_synonyms":
                word_id = component_value["word_id"]
                word = component_value["word"]
                
                syns = get_offline_synonyms(word)
                syns_scores = calculate_synonym_scores(st.session_state.words_data, word_id, syns)
                
                st.session_state.synonyms_payload = {
                    "word_id": word_id,
                    "synonyms_list": syns_scores
                }
                st.rerun()
                
            elif component_value.get("action") == "replace_word":
                word_id = component_value["word_id"]
                new_word = component_value["new_word"]
                st.session_state.words_data[word_id]["word"] = new_word
                st.session_state.words_data[word_id]["isLowPpl"] = False
                
                # Ricalcola le metriche
                new_full_text = " ".join([w["word"] for w in st.session_state.words_data])
                st.session_state.current_ppl = calculate_perplexity(new_full_text)
                st.session_state.current_burst = calculate_burstiness(new_full_text)
                st.rerun()

with col2:
    st.subheader("Auto-Riscrittura Mistral 🧠")
    
    if "rewritten_text" not in st.session_state:
        st.session_state.rewritten_text = ""

    if st.button("🪄 Riscrivi intero testo con Mistral"):
        current_text = " ".join([w["word"] for w in st.session_state.words_data]) if st.session_state.words_data else st.session_state.original_raw_text
        if current_text:
            with st.spinner("Connessione in corso a Mistral API..."):
                st.session_state.rewritten_text = rewrite_with_mistral(current_text)

    edited_text = st.text_area(
        "Modifica manualmente o esporta il risultato:", 
        value=st.session_state.rewritten_text, 
        height=350,
        key="manual_edit"
    )

    if edited_text:
        st.caption("Esporta il risultato:")
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button("⬇️ Scarica .TXT", data=edited_text, file_name="riscrittura.txt", mime="text/plain", use_container_width=True)
        with c_dl2:
            docx_data = create_docx(edited_text)
            st.download_button("⬇️ Scarica .DOCX", data=docx_data, file_name="riscrittura.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
