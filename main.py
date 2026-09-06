import os
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pipeline import generate_palm_reading, ask_palm_question

app = FastAPI(title="Future Vision AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionPayload(BaseModel):
    session_id: str
    question: str
    language: str = "English"

@app.get("/")
async def serve_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "Future Vision AI Online"}

@app.post("/api/analyze-palm")
async def analyze_palm(
    file: UploadFile = File(...),
    language: str = Form("English")
):
    print(f"\n[SERVER] Inbound palm analysis: {file.filename}, Language: {language}")
    try:
        image_bytes = await file.read()
        report, session_id, telemetry = generate_palm_reading(image_bytes, language)
        print(f"[SERVER] Analysis complete for session: {session_id}")
        
        return JSONResponse(content={
            "status": "success",
            "report": report,
            "session_id": session_id,
            "telemetry": telemetry
        })
    except Exception as e:
        print(f"[SERVER ERROR]: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.post("/api/ask-question")
async def ask_question_endpoint(payload: QuestionPayload):
    print(f"\n[SERVER] Follow-up inquiry for session: {payload.session_id}")
    try:
        answer = ask_palm_question(payload.session_id, payload.question, payload.language)
        return JSONResponse(content={
            "status": "success",
            "answer": answer
        })
    except ValueError as ve:
        print(f"[SESSION ERROR]: {ve}")
        return JSONResponse(status_code=400, content={"status": "error", "detail": str(ve)})
    except Exception as e:
        print(f"[SERVER ERROR]: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)