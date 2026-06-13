import streamlit as st
import os
import torch
import difflib
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistralai import Mistral

# Prende la chiave dai Secrets di Hugging Face
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

st.set_page_config(layout="wide", page_title="AI Humanizer")

# --- 1. CARICAMENTO MODELLO LOCALE (Eseguito una sola volta) ---
@st.cache_resource(show_spinner="Scaricamento/Caricamento modello in corso (solo al primo avvio)...")
def load_local_model():
    model_id = "Qwen/Qwen3.5-0.8B"
    # Carichiamo sulla CPU, perfetto per HF Spaces free
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
    return tokenizer, model

tokenizer, model = load_local_model()

# --- 2. FUNZIONI CORE ---
def calculate_perplexity(text):
    """Calcola la perplexity matematica esatta della frase."""
    if not text.strip():
        return 0.0
        
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        
    loss = outputs.loss
    perplexity = torch.exp(loss)
    
    return round(perplexity.item(), 2)

def rewrite_with_mistral(text):
    """Invia il testo a Mistral Small tramite API."""
    if not MISTRAL_API_KEY:
        return "Errore: MISTRAL_API_KEY non trovata nei Secrets."
    
    client = Mistral(api_key=MISTRAL_API_KEY)
    prompt = f"Riscrivi questo testo in modo molto naturale, spezzando il ritmo e rimuovendo i cliché tipici dell'AI. Mantieni il significato:\n\n{text}"
    
    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Errore Mistral API: {e}"

def generate_diff_html(text1, text2):
    """Genera HTML visivo per mostrare le differenze."""
    d = difflib.Differ()
    diffs = list(d.compare(text1.split(), text2.split()))
    
    html = '<div style="font-family: monospace; line-height: 1.6; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">'
    for word in diffs:
        if word.startswith('- '):
            html += f'<span style="background-color: #ffcccc; color: #cc0000; padding: 2px; border-radius: 3px; text-decoration: line-through;">{word[2:]}</span> '
        elif word.startswith('+ '):
            html += f'<span style="background-color: #ccffcc; color: #006600; padding: 2px; border-radius: 3px; font-weight: bold;">{word[2:]}</span> '
        elif word.startswith('  '):
            html += f'<span>{word[2:]}</span> '
    html += '</div>'
    return html

# --- 3. INTERFACCIA UTENTE ---
st.title("🔄 The Humanizer Pipeline")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Testo Originale")
    original_text = st.text_area("Incolla qui il testo sospetto:", height=200)
    
    if st.button("1. Calcola Perplexity Locale"):
        if original_text:
            with st.spinner("Calcolo tensori in corso..."):
                ppl_score = calculate_perplexity(original_text)
                st.info(f"**Score Perplexity:** {ppl_score}")
                st.caption("Valori bassi (es. < 15) = Molto prevedibile/AI. Valori alti = Imprevedibile/Umano.")

with col2:
    st.subheader("Riscrittura & Editing")
    
    if "rewritten_text" not in st.session_state:
        st.session_state.rewritten_text = ""

    if st.button("2. Riscrivi con Mistral Small"):
        if original_text:
            with st.spinner("Connessione a Mistral API..."):
                st.session_state.rewritten_text = rewrite_with_mistral(original_text)

    edited_text = st.text_area(
        "Modifica manualmente se necessario:", 
        value=st.session_state.rewritten_text, 
        height=200,
        key="manual_edit"
    )

st.divider()

st.subheader("Vedi le Differenze (Diff)")
if original_text and edited_text:
    diff_html = generate_diff_html(original_text, edited_text)
    st.markdown(diff_html, unsafe_allow_html=True)
else:
    st.write("Genera la riscrittura per vedere le differenze.")
