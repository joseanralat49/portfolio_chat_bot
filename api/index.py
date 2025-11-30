import os
from typing import Literal, List
from pydantic import BaseModel # type: ignore
from groq import Groq
from fastapi import FastAPI, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from prompt import system_prompt

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://josean-ralat-portfolio.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):  
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]


@app.post("/")
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages array is required")
    
    api_messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.role, "content": m.content} for m in req.messages],
    ]
    
    try: 
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=api_messages,
            temperature=0.3,
        )
        reply = completion.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        print(f"Groq Error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error{e}")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Chatbot API is running"}