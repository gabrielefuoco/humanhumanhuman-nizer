FROM python:3.10-slim

# Crea un utente non-root (Richiesto da Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Imposta le variabili d'ambiente
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Copia i requisiti e installa
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto dell'app
COPY --chown=user . .

# HF Spaces espone di default la porta 7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
