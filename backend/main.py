from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Configura CORS para permitir conexão com o frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_endpoint(query: UserQuery):
    """Endpoint mock que depois será substituído pela LLM"""
    try:
        # Resposta mockada - depois substitua pela chamada real aos agentes
        mock_responses = {
            "olá": "Olá! Como posso te ajudar hoje?",
            "como você está?": "Estou funcionando perfeitamente, obrigado por perguntar!",
            "default": f"Você disse: '{query.query}'. Esta é uma resposta mockada."
        }
        
        response = mock_responses.get(query.query.lower(), mock_responses["default"])
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Servidor está funcionando"}