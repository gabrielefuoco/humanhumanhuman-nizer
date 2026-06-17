import regex as re

def mask_latex(text: str):
    """
    Estrae le strutture LaTeX (formule, commenti, citazioni, comandi strutturali)
    e le sostituisce con maschere testuali (es. [MATH_1]).
    Ritorna il testo mascherato e un dizionario per la ricostruzione.
    """
    registry = {}
    mask_counter = {"BLOCK": 0, "MATH": 0, "CITE": 0, "CMD": 0, "COMMENT": 0}
    
    def replace_with_mask(match, prefix):
        idx = mask_counter[prefix]
        mask_counter[prefix] += 1
        mask_str = f"[{prefix}_{idx}]"
        registry[mask_str] = match.group(0)
        return mask_str

    # 0. Preamble (tutto fino a \begin{document})
    doc_match = re.search(r'\\begin\{document\}', text)
    if doc_match:
        preamble = text[:doc_match.end()]
        idx = mask_counter.setdefault("PREAMBLE", 0)
        mask_counter["PREAMBLE"] += 1
        mask_str = f"[PREAMBLE_{idx}]"
        registry[mask_str] = preamble
        text = mask_str + text[doc_match.end():]

    # 1. Block Math & Comments environments (multiline)
    # Match \begin{equation} ... \end{equation} and similar
    block_patterns = [
        r'\\begin\{(equation|align|eqnarray|gather|multline|comment)\*?\}.*?\\end\{\1\*?\}',
        r'\$\$.*?\$\$',
        r'\\\[.*?\\\]'
    ]
    for pat in block_patterns:
        text = re.sub(pat, lambda m: replace_with_mask(m, "BLOCK"), text, flags=re.DOTALL)
        
    # 2. Line Comments (ignoring escaped \%)
    text = re.sub(r'(?<!\\)%.*', lambda m: replace_with_mask(m, "COMMENT"), text)
    
    # 3. Inline Math
    # $...$ but not $$...$$
    text = re.sub(r'(?<!\\)\$(?!\$).*?(?<!\\)\$', lambda m: replace_with_mask(m, "MATH"), text, flags=re.DOTALL)
    # \(...\)
    text = re.sub(r'\\\(.*?\\\)', lambda m: replace_with_mask(m, "MATH"), text, flags=re.DOTALL)
    
    # 4. Citations & Refs
    text = re.sub(r'\\(?:cite|ref|eqref|label|pageref)\{.*?\}', lambda m: replace_with_mask(m, "CITE"), text)
    
    # 5. Structural Commands and Blocks
    text = re.sub(r'\\begin\{.*?\}', lambda m: replace_with_mask(m, "CMD"), text)
    text = re.sub(r'\\end\{.*?\}', lambda m: replace_with_mask(m, "CMD"), text)
    text = re.sub(r'\\(?:section|subsection|chapter|item|title|author|date|maketitle|tableofcontents)(\*?)(\[.*?\])?\{.*?\}', lambda m: replace_with_mask(m, "CMD"), text)
    
    # Comandi di spaziatura e box con parametri complessi (es. \vspace*{0.4cm}, \makebox[0pt][c]{...})
    text = re.sub(r'\\(?:vspace|hspace|makebox|parbox|rule|setlength|setcounter|addtolength)(\*?)(?:\[.*?\])*(?:\{.*?\})*', lambda m: replace_with_mask(m, "CMD"), text)
    
    # 6. Formatting commands (using recursive regex to handle nested braces)
    text = re.sub(r'\\(?:textbf|textit|emph|underline|textsc|mathrm|mathbf|text|fontsize)(\{(?:[^{}]+|(?1))*\})', lambda m: replace_with_mask(m, "CMD"), text)
    
    # Comandi senza parametri (es. \large, \rmfamily, \selectfont, \par, \noindent)
    text = re.sub(r'\\(?:large|Large|LARGE|huge|Huge|small|footnotesize|scriptsize|tiny|rmfamily|sffamily|ttfamily|mdseries|bfseries|upshape|itshape|slshape|scshape|selectfont|par|noindent|centering|twocolumn|onecolumn)\b', lambda m: replace_with_mask(m, "CMD"), text)
    
    # 7. Altri comandi generici comuni e macro che non sono testo
    text = re.sub(r'\\[a-zA-Z]+\b(?![a-zA-Z\{])', lambda m: replace_with_mask(m, "CMD"), text)

    return text, registry

def unmask_latex(text: str, registry: dict):
    """
    Reinserisce i blocchi LaTeX originali al posto delle maschere.
    """
    if not registry:
        return text
        
    changed = True
    masks = sorted(registry.keys(), key=len, reverse=True)
    while changed:
        changed = False
        for mask in masks:
            if mask in text:
                text = text.replace(mask, registry[mask])
                changed = True
    return text
