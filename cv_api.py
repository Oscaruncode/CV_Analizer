from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import os
import pyodbc

# importa tus funciones ya definidas en el script
from Poc_SpaCy import (
    extract_text_from_pdf,
    extract_text_from_docx,
    load_spacy_model,
    extract_useful_tokens,
    extract_useful_tokens_with_role,
    get_semantic_matches,
    analyze_with_llama,
    infer_role_from_experience,
    extract_score_from_llama
)

app = FastAPI(title="CV Analyzer API", version="1.0")

# -----------------------------
# Función para consultar la BD
# -----------------------------
conn_str = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=DESKTOP-KQ9B2PC;"
    "Database=BD2;"
    "Trusted_Connection=yes;"
)

def get_job_posting_and_requirements(job_id: int):
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Obtener descripción de la vacante
        cursor.execute("SELECT Description FROM JobPostings WHERE JobTitleId = ?", job_id)
        row = cursor.fetchone()
        description = row[0] if row else ""

        # Obtener requerimientos de la vacante
        cursor.execute("""
            SELECT Category, Weight, Keywords, YearsOfExperience
            FROM JobRequirements
            WHERE JobTitleId = ?
        """, job_id)
        requirements = cursor.fetchall()

        # Construimos un texto con keywords y experiencia
        req_text = description + "\n\nRequerimientos:\n"
        for r in requirements:
            category, weight, keywords, years = r
            req_text += f"- {category} (peso {weight}): {keywords} ({years} años)\n"

        conn.close()
        return req_text

    except Exception as e:
        print("❌ Error de conexión:", e)
        return ""

@app.post("/analyze")
async def analyze_cv(
    job_id: int = Form(...),
    cv_file: UploadFile = File(...)
):
    cv_temp = f"temp_{cv_file.filename}"

    try:
        # Guardar temporalmente el archivo CV
        with open(cv_temp, "wb") as f:
            f.write(await cv_file.read())

        # Obtener requerimientos desde la BD
        req_text = get_job_posting_and_requirements(job_id)
        if not req_text:
            return JSONResponse(
                status_code=404,
                content={"error": f"No se encontraron requerimientos para JobId {job_id}"}
            )

        # Extraer texto del CV
        if cv_file.filename.endswith(".pdf"):
            cv_text = extract_text_from_pdf(cv_temp)
        elif cv_file.filename.endswith(".docx"):
            cv_text = extract_text_from_docx(cv_temp)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Formato de CV no soportado. Solo PDF o DOCX."}
            )

        # Procesar con spaCy
        nlp = load_spacy_model(req_text + " " + cv_text)
        req_tokens = extract_useful_tokens(req_text, nlp)
        cv_tokens, years = extract_useful_tokens_with_role(cv_text, nlp)
        role, _ = infer_role_from_experience(cv_text)

        # Matching semántico
        matches, missing = get_semantic_matches(req_tokens, cv_tokens)

        # Análisis con LLaMA
        llama_result = analyze_with_llama(req_text, cv_text, req_tokens, cv_tokens, matches, missing)

        # Score
        score = extract_score_from_llama(llama_result)

        return {
            "jobId": job_id,
            "reqTokens": req_tokens,
            "cvTokens": cv_tokens,
            "yearsExperience": years,
            "roleDetected": role,
            "matches": matches,
            "missing": missing,
            "llamaResult": llama_result,
            "score": score
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(cv_temp):
            os.remove(cv_temp)

@app.get("/get-job/{job_id}")
async def get_job(job_id: int):
    req_text = get_job_posting_and_requirements(job_id)
    if not req_text:
        return {"error": "No se encontró la vacante o hubo un error"}
    return {"jobId": job_id, "requirementsText": req_text}

if __name__ == "__main__":
    uvicorn.run("cv_api:app", host="0.0.0.0", port=8000, reload=True)