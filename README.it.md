# The Humanizer Pipeline

Un'app Gradio che valuta un documento frase per frase sui due segnali che i rilevatori di testo AI approssimano — **perplexity** e **burstiness** — evidenzia le frasi che sembrano scritte da una macchina e le riscrive con modifiche minime e controllate. Costruita su un vincolo forte: il documento è una **tesi in LaTeX**, quindi una riscrittura che rompe una formula, una citazione o un termine tecnico è peggio di nessuna riscrittura.

## Indice

- [Cosa fa](#cosa-fa)
- [Come funziona](#come-funziona)
- [Decisioni](#decisioni)
- [Mappa del repository](#mappa-del-repository)
- [Eseguirla](#eseguirla)
- [Limiti](#limiti)

## Cosa fa

1. **Misura.** Ogni frase riceve una perplexity da un piccolo modello locale, e il documento un valore di burstiness — la deviazione standard della lunghezza delle frasi. Entrambi sono mostrati nell'interfaccia, con le frasi ordinate e paginate.
2. **Individua.** Le frasi sotto la soglia di prevedibilità vengono evidenziate, così si modifica dove conta invece di riscrivere tutto.
3. **Suggerisce.** Per le parole dentro quelle frasi l'app propone sinonimi — da Mistral per scelte consapevoli del contesto, da WordNet come ripiego offline — con un punteggio per candidato.
4. **Riscrive.** La richiesta parte con le frasi vicine come contesto e torna come **modifica minima**: una o due parole sostituite, nient'altro toccato.

## Come funziona

### La misura

La perplexity viene da `Qwen/Qwen3.5-0.8B` in locale tramite `transformers`: la frase è passata sia come ingresso sia come etichetta, e la perplexity è l'esponenziale della loss risultante. È calcolata **per frase**, non per documento, perché l'output utile è una posizione, non un verdetto.

La burstiness è la deviazione standard delle lunghezze delle frasi in parole. La lunghezza uniforme è il segnale più economico di scrittura automatica, e anche il più economico da correggere.

Prima che giri qualsiasi modello, `apply_algorithmic_rules()` intercetta i tic meccanici — trattini lunghi, elenchi di tre, transizioni di repertorio — con espressioni regolari. Una regex è deterministica e gratuita; un modello linguistico non è né l'una né l'altra.

### Proteggere il documento

`latex_parser.py` maschera cinque tipi di struttura prima che succeda qualsiasi cosa: preambolo, blocchi, matematica inline, citazioni e comandi strutturali. Ognuno diventa un segnaposto come `[MATH_3]`, e un registro conserva il testo originale. Tutta la misura e la riscrittura avvengono sul testo mascherato; i segnaposto vengono ripristinati in uscita.

Il prompt di riscrittura impone la stessa regola dal lato del modello: la frase restituita deve contenere ogni segnaposto esattamente come ricevuto, e non deve toccare nomi propri né termini di dominio — *fractional Laplacian*, *overdetermined problem*, *domain*, *singularity*. Un segnaposto mancante è un documento corrotto, e il prompt lo dichiara.

### I vincoli che disegnano l'interfaccia

L'app gira su uno Space di Hugging Face con **ZeroGPU**: il tempo di GPU è concesso in budget fissi (120 s per l'analisi, 60 s per le azioni), ed è per questo che la misura è paginata a **20 frasi per pagina** invece di elaborare una tesi intera in una chiamata. L'ingresso accetta testo incollato, `.docx` o `.tex`, e il risultato si esporta.

## Decisioni

**Prima mascherare, poi riscrivere.** Il disegno ovvio — mandare il testo a un modello e sostituire l'output — distrugge il LaTeX. Mascherare trasforma un problema di generazione aperta in un problema di sostituzione limitata, e rende l'invariante verificabile: quello che è entrato esce, segnaposto compresi.

**Modifiche minime invece di rigenerazione.** Il prompt chiede al massimo una o due sostituzioni di sinonimi e vieta di toccare il resto. Rigenerare riscriverebbe l'argomentazione; una tesi non è un post, e la formulazione dell'autore fa parte della tesi.

**Punteggi per frase invece di un punteggio di documento.** Una perplexity unica non dice dove intervenire. Venti frasi per pagina con un valore ciascuna trasformano la metrica in una lista di cose da fare.

**Contesto nella richiesta di riscrittura.** Le frasi vicine viaggiano con quella da modificare, perché un sinonimo scelto in isolamento è il modo in cui una riscrittura diventa grammaticalmente perfetta e logicamente sconnessa.

**Due fonti di sinonimi con costi diversi.** Mistral quando c'è una chiave, WordNet quando non c'è: l'app continua a funzionare offline, degradata ma non rotta.

**Un modello da 0,8B, non uno grande.** La perplexity richiede la distribuzione di probabilità di un modello linguistico, non la sua intelligenza, e un modello piccolo sta nella RAM dello Space. Il modello che riscrive sta dietro un'API, dove la capacità conta davvero.

## Mappa del repository

```
app.py            interfaccia Gradio, misura, ricerca sinonimi, riscrittura, export
latex_parser.py   maschera e ripristina preambolo, matematica, citazioni, comandi, commenti
main.tex          un documento reale di matematica da 3.258 righe usato come caso di lavoro
requirements.txt  gradio, spaces, transformers, torch, nltk, python-docx, regex
todo.md           note di lavoro: la conversazione di progetto dietro la pipeline
```

## Eseguirla

Su uno Space di Hugging Face: crea uno Space Gradio, pusha questo repository e imposta `MISTRAL_API_KEY` in *Settings → Variables and secrets*. I pesi di Qwen si scaricano al primo avvio; la misura gira sul budget ZeroGPU gratuito e ricade sulla CPU in locale.

In locale:

```bash
pip install -r requirements.txt
export MISTRAL_API_KEY=...
python app.py
```

Senza la chiave funziona tutto tranne la riscrittura e i sinonimi via Mistral: `spaces.GPU` è sostituito da uno stub quando l'app non gira su uno Space.

## Limiti

| Cosa | Impatto | Stato |
|---|---|---|
| La perplexity è un indizio, non un verdetto | dipende dal modello, dalla lingua e dal dominio; un altro modello ordinerebbe diversamente | per progetto |
| Lo strumento ottimizza un proxy | bassa perplexity e alta varianza correlano con i rilevatori, non con la buona scrittura | accettato, dichiarato |
| Solo italiano e inglese | prompt e guardie sono scritti in queste due lingue | aperto |
| Le maschere LaTeX sono posizionali | una riscrittura che riordina la frase attorno a un segnaposto può produrre output strano | aperto |
| La misura sull'intero documento è paginata | una tesi richiede diversi clic, imposto dal budget ZeroGPU | accettato |
| Nessun test automatico | gli invarianti si controllano a occhio sul documento esportato | aperto |

Una nota che sta nel README e non in un messaggio di commit: l'uso onesto di questo strumento è riscrivere un proprio testo perché si legga meglio. Ottimizzare contro un rilevatore significa ottimizzare contro un proxy, e il proxy è debole in entrambe le direzioni — segnala come scritte da una macchina le prose di autori non madrelingua e di classici, e accetta come umano un testo automatico corretto a mano.
