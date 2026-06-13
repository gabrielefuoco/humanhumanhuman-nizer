import streamlit as st
import os
import torch
import nltk
import re
import math
from io import BytesIO
from docx import Document
from nltk.corpus import wordnet as wn
import streamlit.components.v1 as components
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from latex_parser import mask_latex, unmask_latex

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
    .metric-box { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); margin-top: 10px; text-align: center;}
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
Sei un editor umano esperto nel riscrivere testi generati dall'AI per renderli al 100% indistinguibili dalla vera scrittura umana. 
Il tuo unico scopo è riscrivere la frase o il paragrafo fornito abbassando drasticamente la sua prevedibilità matematica (aumentando la perplexity) e simulando l'imprevedibilità stilistica di un autore umano (burstiness).

REGOLE FONDAMENTALI:
1. EVITA IL LESSICO DA AI: Non usare MAI espressioni banali, pompose o tipiche dei modelli linguistici come "fondamentale", "cruciale", "complesso", "un quadro", "svolge un ruolo", "in sintesi", "è importante notare", "tuttavia", "inoltre".
2. LESSICO IMPREVEDIBILE: Usa sinonimi meno ovvi e più vari. Se una parola è troppo comune, trovala una meno prevedibile ma altrettanto corretta.
3. ALTERA LA SINTASSI: Evita la classica struttura "Soggetto + verbo + complemento". Inverti l'ordine delle parole, spezza le frasi lunghe in frasi più brevi, oppure usa incisi e virgole per spezzare il ritmo meccanico. Passa dalla forma passiva a quella attiva dove possibile.
4. "SPORCA" LO STILE: La scrittura umana è leggermente asimmetrica e meno "perfetta" di quella dell'AI. Cerca un tono più diretto, meno enciclopedico e leggermente più discorsivo, mantenendo però la correttezza grammaticale.
5. PRESERVA IL SIGNIFICATO: Non inventare fatti e non omettere informazioni chiave. Il senso originale deve rimanere identico.
6. PRESERVA I SEGNAPOSTO (CRITICO): DEVI conservare intatti e nella giusta posizione tutti i placeholder come [MATH_n], [CITE_n], [CMD_n]. Non tradurli o modificarli per nessun motivo.

OUTPUT:
Restituisci ESCLUSIVAMENTE la frase riscritta. Nessuna introduzione, nessuna spiegazione, nessuna virgoletta iniziale o finale.
"""

def apply_algorithmic_rules(text):
    if not text: return text
    text = text.replace("—", ", ").replace("–", ", ")
    text = text.replace(" — ", ", ").replace(" -- ", ", ")
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = text.replace("“", '"').replace("”", '"')
    return text

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]

def calculate_burstiness(sentences_data):
    if not sentences_data: return 0.0
    lengths = [len(s["words"]) for s in sentences_data]
    if len(lengths) < 2: return 0.0
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return round(math.sqrt(variance), 2)

def process_sentence(sentence_text):
    """Calcola perplexity della frase e dei singoli token."""
    qwen_text = sentence_text
    is_latex = st.session_state.get("is_latex", False)
    if is_latex:
        for mask in st.session_state.latex_registry.keys():
            qwen_text = qwen_text.replace(mask, "formula")
            
    clean_qwen = qwen_text.replace("formula", "").strip()
    if is_latex and not clean_qwen:
        words_data = []
        for w in sentence_text.split():
            word_str = w
            is_mask = False
            for mask, real_val in st.session_state.latex_registry.items():
                if mask in word_str:
                    word_str = word_str.replace(mask, real_val)
                    is_mask = True
            words_data.append({
                "word": word_str,
                "isLowPpl": False,
                "isCliche": False,
                "isMask": is_mask
            })
        return {
            "text": sentence_text,
            "words": words_data,
            "isCritical": False
        }

    words_data = []
    encodings = tokenizer(qwen_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings.input_ids[0]
    
    try:
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), labels=input_ids.unsqueeze(0))
        logits = outputs.logits[0, :-1, :]
        labels = input_ids[1:]
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(logits, labels).tolist()
        losses = [0.0] + losses
        avg_loss = sum(losses)/len(losses) if losses else 0
    except Exception:
        losses = [10.0] * len(input_ids)
        avg_loss = 10.0
        
    words = sentence_text.split()
    token_idx = 0
    for w in words:
        w_tokens = tokenizer(w, add_special_tokens=False).input_ids
        num_tokens = max(1, len(w_tokens))
        
        w_losses = []
        for _ in range(num_tokens):
            if token_idx < len(losses):
                w_losses.append(losses[token_idx])
                token_idx += 1
                
        word_loss = sum(w_losses) / len(w_losses) if w_losses else 10.0
            
        try:
            ppl = math.exp(word_loss)
        except OverflowError:
            ppl = 1000.0
            
        clean_w = "".join(c for c in w if c.isalpha())
        is_low_ppl = (ppl < 15.0 and len(clean_w) > 3)
        is_cliche = clean_w.lower() in AI_CLICHES
        
        words_data.append({
            "word": w,
            "isLowPpl": is_low_ppl,
            "isCliche": is_cliche
        })
        
    sentence_ppl = round(math.exp(avg_loss) if avg_loss < 100 else 1000.0, 1)
    cliche_count = sum(1 for w in words_data if w["isCliche"])
    is_critical = (sentence_ppl < 30.0) or (cliche_count > 0)
    
    # Calcolo approssimativo di AI Detection Score (0-100%)
    ai_score = round(max(0, min(100, 100 - (sentence_ppl * 2) + (cliche_count * 10))))
    
    if is_latex:
        for w_dict in words_data:
            for mask, real_val in st.session_state.latex_registry.items():
                if mask in w_dict["word"]:
                    w_dict["word"] = w_dict["word"].replace(mask, real_val)
                    w_dict["isMask"] = True
                    w_dict["isLowPpl"] = False
                    w_dict["isCliche"] = False
    
    return {
        "text": sentence_text,
        "words": words_data,
        "isCritical": is_critical,
        "ppl": sentence_ppl,
        "aiScore": ai_score
    }

def get_mistral_synonyms(word, context_sentence):
    if not MISTRAL_API_KEY: return []
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    data = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": "Sei un dizionario dei sinonimi. Restituisci SOLO una lista di 5 sinonimi per la parola richiesta, separati da virgola. Nessun'altra parola o punteggiatura extra."},
            {"role": "user", "content": f"Fornisci 5 sinonimi per la parola '{word}' nel contesto di questa frase: '{context_sentence}'"}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        raw_output = response.json()["choices"][0]["message"]["content"]
        return [s.strip() for s in raw_output.split(",") if s.strip()]
    except:
        return []

def get_offline_synonyms(word, context_sentence=""):
    clean_word = "".join(c for c in word if c.isalpha()).lower()
    synsets = wn.synsets(clean_word, lang='ita')
    syns = set()
    for syn in synsets:
        for lemma in syn.lemma_names('ita'):
            if lemma.lower() != clean_word:
                syns.add(lemma.replace('_', ' '))
    syns_list = list(syns)[:5]
    if not syns_list and context_sentence:
        syns_list = get_mistral_synonyms(clean_word, context_sentence)
    return syns_list

def calculate_synonym_scores(sentence_data, word_idx, synonyms):
    results = []
    left_context = " ".join([w["word"] for w in sentence_data["words"][:word_idx]])
    right_context = " ".join([w["word"] for w in sentence_data["words"][word_idx+1:]])
    
    for syn in synonyms:
        new_text = f"{left_context} {syn} {right_context}".strip()
        encodings = tokenizer(new_text, return_tensors="pt", truncation=True, max_length=512)
        try:
            with torch.no_grad():
                outputs = model(encodings.input_ids, labels=encodings.input_ids)
            score = math.exp(outputs.loss.item())
        except:
            score = 1000.0
        results.append({"word": syn, "score": score})
        
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def rewrite_with_mistral(text, sentence_data=None):
    if not MISTRAL_API_KEY:
        return text
    
    prompt = f"Riscrivi la seguente frase problematica in modo che sembri 100% scritta da un essere umano. Testo:\n\n{text}"
    
    if sentence_data:
        suggestions = []
        for i, w_dict in enumerate(sentence_data["words"]):
            if w_dict.get("isLowPpl") or w_dict.get("isCliche"):
                syns = get_offline_synonyms(w_dict["word"], text)
                if syns:
                    scores = calculate_synonym_scores(sentence_data, i, syns)
                    if scores:
                        top_syns = ", ".join([f"{s['word']} (PPL: {round(s['score'], 1)})" for s in scores[:3]])
                        suggestions.append(f"- {w_dict['word']}: {top_syns}")
        if suggestions:
            prompt += "\n\nPer aiutarti a bypassare i detector AI, ecco dei sinonimi ad altissima perplexity raccomandati per sostituire le parole incriminate. Cerca di usarli nel testo in modo naturale e grammaticalmente coerente:\n" + "\n".join(suggestions)
            
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    data = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_HUMANIZER},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        raw_output = response.json()["choices"][0]["message"]["content"]
        return apply_algorithmic_rules(raw_output)
    except Exception as e:
        return text

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
if "processed_sentences" not in st.session_state:
    st.session_state.processed_sentences = []
if "to_process" not in st.session_state:
    st.session_state.to_process = []
if "synonyms_payload" not in st.session_state:
    st.session_state.synonyms_payload = None
if "last_action_id" not in st.session_state:
    st.session_state.last_action_id = None

# --- UI PRINCIPALE ---
st.title("✨ The Humanizer Pipeline")
st.markdown("Scrivi o carica un testo. Le frasi verranno processate in streaming. Fai click su **Riscrivi** per correggere le frasi sospette o usa i sinonimi per nascondere il testo dai detector AI.")

# --- AREA DI INPUT ---
if not st.session_state.processed_sentences and not st.session_state.to_process:
    st.subheader("Carica il tuo testo")
    uploaded_file = st.file_uploader("📂 Carica un documento (TXT, MD, DOCX, TEX)", type=["txt", "md", "docx", "tex"])
    
    default_text = ""
    if uploaded_file is not None:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            default_text = extract_text_from_file(uploaded_file)
            st.session_state.last_uploaded_file = uploaded_file.name

    input_text = st.text_area("Oppure incolla qui il testo e clicca 'Analizza'", value=default_text, height=250, key="input_raw")
    
    if st.button("🔍 Avvia Analisi Streaming", use_container_width=True):
        if input_text:
            is_latex = False
            latex_registry = {}
            raw_text_for_export = input_text
            
            if st.session_state.get("last_uploaded_file", "").endswith(".tex"):
                is_latex = True
                input_text, latex_registry = mask_latex(input_text)
                raw_text_for_export = input_text # Salviamo il testo mascherato come base
                
            st.session_state.is_latex = is_latex
            st.session_state.latex_registry = latex_registry
            st.session_state.raw_text_for_export = raw_text_for_export
            # Copia di backup per tracciare le modifiche fatte da Mistral durante l'export
            st.session_state.original_sentences_map = {}

            st.session_state.to_process = split_into_sentences(input_text)
            st.session_state.processed_sentences = []
            st.rerun()

# --- LOOP DI STREAMING (1 frase alla volta per non bloccare il server) ---
needs_rerun = False
if len(st.session_state.to_process) > 0:
    total_sentences = len(st.session_state.to_process) + len(st.session_state.processed_sentences)
    current = len(st.session_state.processed_sentences) + 1
    
    st.markdown(f"⏳ *Elaborazione in corso... Frase {current} di {total_sentences}*")
    progress_bar = st.progress(current / total_sentences)
    
    # Processa UNA SOLA frase
    sentence_text = st.session_state.to_process.pop(0)
    s_data = process_sentence(sentence_text)
    s_data["original_text"] = sentence_text
    st.session_state.processed_sentences.append(s_data)
    
    needs_rerun = True

# --- EDITOR E METRICHE ---
if st.session_state.processed_sentences:
    # Metriche Globali
    burstiness = calculate_burstiness(st.session_state.processed_sentences)
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>Frasi Analizzate</div><div class='metric-value'>{len(st.session_state.processed_sentences)}</div></div>", unsafe_allow_html=True)
    with c_m2:
        color_class = "metric-human" if burstiness > 5 else "metric-ai"
        st.markdown(f"<div class='metric-box'><div class='metric-title'>Burstiness Globale</div><div class='metric-value {color_class}'>{burstiness}</div></div>", unsafe_allow_html=True)
    
    st.divider()

    # Componente Frontend
    payload = st.session_state.synonyms_payload
    st.session_state.synonyms_payload = None # Consumato
    
    component_value = interactive_text(
        sentences=st.session_state.processed_sentences,
        synonyms_update=(payload is not None),
        synonyms_s_idx=payload["s_idx"] if payload else None,
        synonyms_w_idx=payload["w_idx"] if payload else None,
        synonyms_list=payload["syns_scores"] if payload else None,
        key="interactive_editor"
    )

    # GESTIONE EVENTI DA JS
    if component_value:
        action_id = str(component_value)
        if st.session_state.last_action_id != action_id:
            st.session_state.last_action_id = action_id
            action = component_value.get("action")
            
            if action == "get_synonyms":
                s_idx = component_value["sentence_idx"]
                w_idx = component_value["word_idx"]
                word = component_value["word"]
                context_sentence = st.session_state.processed_sentences[s_idx]["text"]
                
                syns = get_offline_synonyms(word, context_sentence)
                syns_scores = calculate_synonym_scores(st.session_state.processed_sentences[s_idx], w_idx, syns)
                
                st.session_state.synonyms_payload = {
                    "s_idx": s_idx,
                    "w_idx": w_idx,
                    "syns_scores": syns_scores
                }
                st.rerun()
                
            elif action == "replace_word":
                s_idx = component_value["sentence_idx"]
                w_idx = component_value["word_idx"]
                new_word = component_value["new_word"]
                
                s = st.session_state.processed_sentences[s_idx]
                old_word = s["words"][w_idx]["word"]
                new_text = s["text"].replace(old_word, new_word, 1)
                
                original_text = s.get("original_text", s["text"])
                reprocessed = process_sentence(new_text)
                reprocessed["original_text"] = original_text
                st.session_state.processed_sentences[s_idx] = reprocessed
                st.rerun()
                
            elif action == "rewrite_sentence":
                s_idx = component_value["sentence_idx"]
                old_text = component_value["text"]
                
                # Riscrivi con Mistral passandogli i dati della frase
                s_data = st.session_state.processed_sentences[s_idx]
                new_text = rewrite_with_mistral(old_text, sentence_data=s_data)
                
                # Processa e rimpiazza in-place
                original_text = st.session_state.processed_sentences[s_idx].get("original_text", old_text)
                reprocessed = process_sentence(new_text)
                reprocessed["original_text"] = original_text
                st.session_state.processed_sentences[s_idx] = reprocessed
                st.rerun()

    st.divider()
    
    # Pulsanti di Export
    full_text = " ".join([s["text"] for s in st.session_state.processed_sentences])
    c_dl1, c_dl2, c_dl3 = st.columns(3)
    
    is_latex = st.session_state.get("is_latex", False)
    
    with c_dl1:
        if is_latex:
            tex_text = st.session_state.raw_text_for_export
            for s in st.session_state.processed_sentences:
                if s.get("original_text") and s["original_text"] != s["text"]:
                    tex_text = tex_text.replace(s["original_text"], s["text"])
            tex_text = unmask_latex(tex_text, st.session_state.latex_registry)
            st.download_button("⬇️ Scarica .TEX", data=tex_text, file_name="riscrittura.tex", mime="text/plain", use_container_width=True)
        else:
            st.download_button("⬇️ Scarica .TXT", data=full_text, file_name="riscrittura.txt", mime="text/plain", use_container_width=True)
    with c_dl2:
        docx_data = create_docx(full_text)
        st.download_button("⬇️ Scarica .DOCX", data=docx_data, file_name="riscrittura.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
    with c_dl3:
        if st.button("🔄 Ricomincia da capo", use_container_width=True):
            st.session_state.processed_sentences = []
            st.session_state.to_process = []
            st.rerun()

# Chiama il rerun alla fine per far aggiornare il frontend e poi ripartire
if needs_rerun:
    st.rerun()
