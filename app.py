import streamlit as st
import os
import torch
import difflib
import nltk
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

# --- FUNZIONI CORE ---
def calculate_perplexity(text):
    if not text.strip(): return 0.0
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return round(torch.exp(outputs.loss).item(), 2)

def analyze_text_token_by_token(text):
    """Calcola la perplexity per le singole parole usando un allineamento semplificato."""
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
    
    # Allineamento euristico semplice 1 parola = 1 o più token
    # Qui simuliamo la mappatura leggendo progressivamente le loss
    token_idx = 0
    for w in text_words:
        word_loss = 10.0
        if token_idx < len(losses):
            word_loss = losses[token_idx]
            token_idx += 1
            
        ppl = torch.exp(torch.tensor(word_loss)).item()
        
        # Una parola è considerata "Ai-like" se è molto prevedibile (< 15) e lunga
        clean_w = "".join(c for c in w if c.isalpha())
        is_low_ppl = (ppl < 15.0 and len(clean_w) > 3)
        
        words_data.append({
            "word": w,
            "isLowPpl": is_low_ppl
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
    prompt = f"Riscrivi in modo naturale e umano. Alterna frasi corte e lunghe. Evita cliché. Testo:\n{text}"
    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Errore Mistral API: {e}"

# --- INIZIALIZZAZIONE STATO ---
if "words_data" not in st.session_state:
    st.session_state.words_data = []
if "original_raw_text" not in st.session_state:
    st.session_state.original_raw_text = ""

# --- UI PRINCIPALE ---
st.title("✨ The Humanizer Pipeline")
st.markdown("Analizza la prevedibilità statistica del testo (*Perplexity*) parola per parola e sostituisci interattivamente i cliché AI con sinonimi dal dizionario NLTK.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Editor Interattivo 📝")
    st.caption("Fai **Click Destro** sulle parole rosse per aprire il dizionario dei sinonimi NLTK.")
    
    # Campo di input per il testo iniziale
    input_text = st.text_area("Incolla qui il testo sospetto e clicca 'Analizza'", height=150, key="input_raw")
    
    if st.button("🔍 Analizza Token-by-Token"):
        if input_text:
            with st.spinner("Analisi dei tensori in corso..."):
                st.session_state.original_raw_text = input_text
                st.session_state.words_data = analyze_text_token_by_token(input_text)
                st.rerun()
    
    st.divider()
    
    # Render del componente custom solo se ci sono dati
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
            
        # Gestione eventi dal frontend custom
        if component_value:
            if component_value.get("action") == "get_synonyms":
                word_id = component_value["word_id"]
                word = component_value["word"]
                
                # Calcola offline i sinonimi e i loro score PPL nel contesto
                syns = get_offline_synonyms(word)
                syns_scores = calculate_synonym_scores(st.session_state.words_data, word_id, syns)
                
                # Salviamo il payload per aggiornare il frontend
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
                st.rerun()

with col2:
    st.subheader("Auto-Riscrittura Mistral 🧠")
    
    if "rewritten_text" not in st.session_state:
        st.session_state.rewritten_text = ""

    if st.button("🪄 Riscrivi intero testo con Mistral"):
        # Ricostruiamo il testo da mandare a Mistral (prende l'ultima versione dell'editor interattivo)
        current_text = " ".join([w["word"] for w in st.session_state.words_data]) if st.session_state.words_data else st.session_state.original_raw_text
        if current_text:
            with st.spinner("Connessione in corso a Mistral API..."):
                st.session_state.rewritten_text = rewrite_with_mistral(current_text)

    edited_text = st.text_area(
        "Modifica manualmente il risultato se necessario:", 
        value=st.session_state.rewritten_text, 
        height=350,
        key="manual_edit"
    )
