from fastapi import APIRouter
from typing import List
import time, asyncio, random, json, csv
from io import StringIO
import httpx
from models import AnalysisRequest, SentimentOutput

router = APIRouter(prefix="/sentiment", tags=["Sentiment Analysis"])

# Configuration constants
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
BATCH_SIZE = 10  # Maximum number of responses processed per batch
MAX_RETRIES = 3   # Maximum number of retries for each batch
RETRY_DELAY = 3   # Delay (in seconds) before retrying a failed batch

def build_prompt(responses_batch: list) -> str:
    prompt = f"""
Eres un analista experto en encuestas de clima laboral.
Tu única tarea es clasificar cada respuesta en un valor numérico de sentimiento.
No expliques, no agregues texto extra, no agregues comentarios: devuelve únicamente CSV válido.

Criterios de clasificación:
0 = Negativo
1 = Positivo
2 = Neutro

Reglas estrictas de salida:
- La salida debe ser SOLO un CSV válido.
- El CSV debe usar saltos de línea reales, no literales "\\n" ni "\\r\\n".
- Usa coma (,) como separador, sin espacios adicionales.
- Incluye la primera fila como encabezados: AnswerID,Sentiment
- Una fila por cada respuesta, en el mismo orden recibido.
- No agregues texto antes ni después del CSV.

Ejemplo de salida esperada:
AnswerID,Sentiment
101,1
102,0
103,2

ANALIZA EL SIGUIENTE JSON:
{json.dumps([resp.dict() for resp in responses_batch], ensure_ascii=False, separators=(',', ':'))}
"""
    return prompt.strip()

async def call_ollama_with_retry(prompt: str, batch_index: int, batch_size: int) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n➡️ Sending batch {batch_index} (attempt {attempt}) with {batch_size} responses...")

            # Deterministic config by default
            options = {
                "temperature": 0.1,
                "repeat_penalty": 1.0,
                "seed": 42
            }

            # On retries, we loosen up
            if attempt > 1:
                options["temperature"] = 1
                options["repeat_penalty"] = 1.2
                options["seed"] = random.randint(1, 1_000_000)

            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=260.0) as client:
                response = await client.post(
                    OLLAMA_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": options
                    },
                )
            elapsed = time.perf_counter() - start
            print(f"✅ Batch {batch_index} responded in {elapsed:.2f} seconds")
            print(f"📏 Response size: {len(response.text)} characters")
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except Exception as e:
            print(f"⚠️ Error in batch {batch_index} (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                print(f"🔁 Retrying in {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"❌ Batch {batch_index} failed after {MAX_RETRIES} attempts.") from e


async def call_and_validate_batch(batch, batch_index: int) -> List[SentimentOutput]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt = build_prompt(batch)
            raw_output = await call_ollama_with_retry(prompt, batch_index, len(batch))

            # Remove empty lines from the response
            raw_output = "\n".join([line for line in raw_output.splitlines() if line.strip()])

            # Parse CSV output
            f = StringIO(raw_output)
            reader = csv.reader(f)
            rows = list(reader)

            if not rows:
                raise ValueError("CSV vacío")

            # Detect header row
            first_row = rows[0]
            if first_row == ["AnswerID", "Sentiment"]:
                data_rows = rows[1:]
            elif all(cell.isdigit() for cell in first_row[1:]):
                data_rows = rows
            else:
                data_rows = rows

            # Validate row format and sentiment values
            results = []
            for row in data_rows:
                if len(row) != 2:
                    raise ValueError(f"Invalid row format: {row}")
                answer_id, sentiment_str = row
                if sentiment_str not in {"0", "1", "2"}:
                    raise ValueError(f"Invalid sentiment in AnswerID {answer_id}: {sentiment_str}")
                results.append(SentimentOutput(AnswerID=answer_id, Sentiment=int(sentiment_str)))

            return results

        except ValueError as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                raise

@router.post("/analyze", response_model=List[SentimentOutput])
async def analyze_sentiments(payload: AnalysisRequest):
    start_total = time.perf_counter()

    # Split responses into batches
    batches = [
        payload.responses[i:i + BATCH_SIZE]
        for i in range(0, len(payload.responses), BATCH_SIZE)
    ]
    print(f"\n📊 Total responses: {len(payload.responses)}")
    print(f"📦 Total batches: {len(batches)}, batch size: {BATCH_SIZE}\n")

    results = []
    for i, batch in enumerate(batches):
        try:
            batch_results = await call_and_validate_batch(batch, batch_index=i + 1)
            results.extend(batch_results)
            print(f"✅ Batch {i + 1} processed, parsed responses: {len(batch_results)}")
        except Exception as e:
            print(f"❌ Batch {i + 1} failed after {MAX_RETRIES} attempts. Skipping. Error: {e}")
            continue    # Skip failed batch and continue processing

    elapsed_total = time.perf_counter() - start_total
    print(f"\n⏱️ Total analysis time for {len(payload.responses)} responses: {elapsed_total:.2f} seconds")
    return results