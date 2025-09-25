from PyPDF2 import PdfReader
from docx import Document
import spacy
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
import requests
import re
from datetime import datetime, UTC
from dateutil import parser as dtparser
import time

# -----------------------------
# Configuración
# -----------------------------
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# -----------------------------
# Extraer texto de archivos
# -----------------------------
def load_requirements(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return " ".join([p.text for p in doc.paragraphs])

def load_cv(path: str) -> str:
    if path.endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif path.endswith(".docx"):
        return extract_text_from_docx(path)
    else:  # fallback a txt
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

# -----------------------------
# Cargar modelo spaCy según idioma
# -----------------------------
def load_spacy_model(text: str):
    lang = detect(text)
    print(f"📌 Idioma detectado: {lang}")
    if lang == "es":
        return spacy.load("es_core_news_md")
    else:
        return spacy.load("en_core_web_md")

# -----------------------------
# Experiencia laboral: rangos + menciones
# -----------------------------
def years_from_mentions(text):
    pat = re.compile(r'(\d+(?:[.,]\d+)?)(?=\s*\+?)\s*(años|anos|year|years|yr|yrs)', re.IGNORECASE)
    vals = [float(n.replace(",", ".")) for n,_ in pat.findall(text)]
    return sum(vals) if vals else 0.0

def parse_fuzzy_date(s, anchor='start'):
    s = s.strip()
    if re.search(r'present|actualidad|now|hoy', s, re.IGNORECASE):
        return datetime.now(UTC)

    if re.fullmatch(r'\d{4}', s):
        y = int(s); m = 1 if anchor=='start' else 12
        d = 1 if anchor=='start' else 28
        return datetime(y,m,d)

    if re.fullmatch(r'\d{1,2}[/-]\d{4}', s):
        m,y = re.split(r'[/-]', s)
        m=int(m); y=int(y); d=1 if anchor=='start' else 28
        return datetime(y,m,d)
    return dtparser.parse(s, default=datetime(2000,1,1))

def iter_ranges(text):
    text = re.sub(r'[\u2013\u2014\u2212]', '-', text)
    rgx = re.compile(
        r'(?P<s>(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
        r'|Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic)\b\.?\s+\d{4}'
        r'|\b\d{1,2}[/-]\d{4}|\b\d{4}))\s*(?:to|a|hasta|-|–)\s*'
        r'(?P<e>(?:Present|Actualidad|Now|Hoy'
        r'|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
        r'|Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic)\b\.?\s+\d{4}'
        r'|\b\d{1,2}[/-]\d{4}|\b\d{4}))',
        re.IGNORECASE
    )
    for m in rgx.finditer(text):
        try:
            s = parse_fuzzy_date(m.group('s'), 'start')
            e = parse_fuzzy_date(m.group('e'), 'end')
            if s and e and e >= s:
                yield s, e
        except:
            continue

def months_between(s,e): 
    return (e.year-s.year)*12+(e.month-s.month)+1

def years_from_ranges(text):
    months = sum(max(0, months_between(s,e)) for s,e in iter_ranges(text))
    return round(months/12.0, 2)

def infer_role_from_experience(text: str):
    y_lit  = years_from_mentions(text)
    y_rng  = years_from_ranges(text)
    years  = max(y_lit, y_rng)

    if years < 2:
        role = "trainee"
    elif 2 <= years < 4:
        role = "junior"
    elif 4 <= years < 7:
        role = "semi-senior"
    else:
        role = "senior"

    return role, years

# -----------------------------
# Limpieza y extracción de tokens útiles
# -----------------------------
def extract_useful_tokens(text: str, nlp):
    doc = nlp(text)
    tokens = []
    custom_stop = {"summary", "responsible", "skill", "experience", "team", "project"}

    for token in doc:
        txt = token.lemma_.lower().strip()
        if (
            not token.is_stop
            and not token.is_punct
            and not token.like_num
            and token.ent_type_ not in {"PERSON", "GPE", "DATE", "TIME", "MONEY", "EMAIL", "PHONE"}
            and len(txt) > 2
            and not re.match(r"^\W+$", txt)
            and not txt.startswith("-")
            and not re.match(r"^[\n\r\s]+$", txt)
            and not re.match(r".*@.*", txt)
            and not re.match(r"\d{3,}", txt)
            and not re.match(r"http[s]?://", txt)
            and txt not in custom_stop
        ):
            tokens.append(txt)

    return sorted(set(tokens))

# -----------------------------
# Añade rol + años de experiencia a los tokens
# -----------------------------
def extract_useful_tokens_with_role(text: str, nlp):
    tokens = extract_useful_tokens(text, nlp)
    role, years = infer_role_from_experience(text)

    if role and role not in tokens:
        tokens.append(role)

    tokens.append(f"{int(years)}_years")
    return sorted(set(tokens)), years

# -----------------------------
# Matching semántico con embeddings
# -----------------------------
def get_semantic_matches(req_tokens, cv_tokens, threshold=0.7, soft_threshold=0.6):
    matches, missing = [], []
    if not req_tokens or not cv_tokens:
        return matches, req_tokens

    req_emb = embedder.encode(req_tokens, convert_to_tensor=True)
    cv_emb = embedder.encode(cv_tokens, convert_to_tensor=True)

    for i, req in enumerate(req_tokens):
        sim_scores = util.cos_sim(req_emb[i], cv_emb)[0]
        best_idx = int(sim_scores.argmax())
        best_score = float(sim_scores[best_idx])

        if best_score >= threshold:
            matches.append((req, cv_tokens[best_idx], round(best_score, 3)))
        elif best_score >= soft_threshold:
            # Acepta "coincidencia difusa"
            matches.append((req, cv_tokens[best_idx], round(best_score, 3)))
        else:
            missing.append(req)

    return matches, missing

# -----------------------------
# Análisis con LLaMA 3
# -----------------------------
def analyze_with_llama(req_text, cv_text, req_tokens, cv_tokens, matches, missing):
    prompt = f"""
Eres un sistema experto en reclutamiento.
Analiza la compatibilidad entre esta vacante y el candidato.

📌 Resumen de requerimientos: {req_tokens}
📌 Resumen del CV: {cv_tokens}
📌 Coincidencias: {len(matches)} / {len(req_tokens)}
📌 Requisitos posiblemente faltantes: {missing}

Con base en esto:
1. Da un puntaje global de compatibilidad (0 a 100).
2. Explica de forma general qué tan bien el candidato cumple los requisitos técnicos y de experiencia (no detalle por requisito).
3. Da una conclusión final: ¿apto o no para el rol?
4. Ten en cuenta que algunos tokens pueden estar escritos de forma diferente en el CV
"""

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()
    return result.get("response", "")

def extract_score_from_llama(text: str) -> int:
    """
    Busca un puntaje en el formato '85/100', '85 de 100' o 'puntaje: 85'
    y lo devuelve como entero. Si no encuentra nada, devuelve 0.
    """
    if not text:
        return 0

    # Caso "85/100" o "85 / 100"
    match = re.search(r"(\d{1,3})\s*[/ ]\s*100", text)
    if match:
        return min(100, int(match.group(1)))

    # Caso "85 de 100"
    match = re.search(r"(\d{1,3})\s*de\s*100", text, re.IGNORECASE)
    if match:
        return min(100, int(match.group(1)))

    # Caso "puntaje: 85" o "compatibilidad 85"
    match = re.search(r"(puntaje|compatibilidad|score)\D*(\d{1,3})", text, re.IGNORECASE)
    if match:
        return min(100, int(match.group(2)))

    return 0

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    total_start = time.perf_counter()

    t0 = time.perf_counter()
    req_path = r"C:\Python\POC - CV\requirements.txt"
    cv_path = r"C:\Python\POC - CV\CV_Wilmar_Alfaro.pdf"

    req_text = load_requirements(req_path)
    cv_text = load_cv(cv_path)
    print(f"⏱ Carga archivos: {time.perf_counter() - t0:.2f} s")

    t0 = time.perf_counter()
    nlp = load_spacy_model(req_text + " " + cv_text)
    print(f"⏱ Carga modelo spaCy: {time.perf_counter() - t0:.2f} s")

    t0 = time.perf_counter()
    req_tokens = extract_useful_tokens(req_text, nlp)
    cv_tokens, years = extract_useful_tokens_with_role(cv_text, nlp)
    role, _ = infer_role_from_experience(cv_text)

    print("🔹 Tokens requisitos:", req_tokens)
    print("🔹 Tokens CV:", cv_tokens)
    print(f"🔹 Años de experiencia acumulados: {years}")
    print(f"🔹 Rol detectado: {role}")
    print(f"⏱ Extracción tokens: {time.perf_counter() - t0:.2f} s")

    t0 = time.perf_counter()
    matches, missing = get_semantic_matches(req_tokens, cv_tokens)
    print("\n✅ MATCHING SEMÁNTICO")
    print("Coincidencias:", matches)
    print("Faltantes:", missing)
    print(f"⏱ Matching embeddings: {time.perf_counter() - t0:.2f} s")

    t0 = time.perf_counter()
    llama_result = analyze_with_llama(req_text, cv_text, req_tokens, cv_tokens, matches, missing)
    score = extract_score_from_llama(llama_result)
    print("\n🤖 RESULTADO LLaMA 3:\n", llama_result)
    print(f"📊 Score extraído dinámicamente: {score}")
    print(f"⏱ LLaMA análisis: {time.perf_counter() - t0:.2f} s")

    print(f"✅ Tiempo total ejecución: {time.perf_counter() - total_start:.2f} s")