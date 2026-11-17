"""
Universal High-Throughput Serving Server for QJev-1.5B.
Supports:
1. POST /v1/systemone         (Official JevBench & BenchmarkHeaven protocol)
2. POST /v1/decide            (Simple, developer-friendly REST endpoint)
3. POST /v1/chat/completions  (OpenAI-compatible drop-in endpoint)
4. GET  /health               (Kubernetes / Docker health check)

Usage:
    python serve.py --port 8000 --model-path export/QJev-1.5B-Phase3
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

repo_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, repo_root)
from qjev import QJev

app = FastAPI(
    title="QJev System 1 Decision Model Server",
    description="Universal inference server for QJev-1.5B supporting System One, OpenAI, and REST protocols.",
    version="1.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_instance: Optional[QJev] = None

# --- Request / Response Schemas ---
class DecideRequest(BaseModel):
    context: str = Field(..., description="The situation or context text to evaluate.")
    choices: List[str] = Field(..., description="List of options (up to 10).")
    question: Optional[str] = Field(None, description="Optional custom question prompt.")
    temperature: Optional[float] = Field(1.0, description="Softmax calibration temperature.")

class DecideResponse(BaseModel):
    best_choice: str
    confidence: float
    probabilities: Dict[str, float]

class SystemOneQuestion(BaseModel):
    type: str = Field(..., description="'choice', 'noul', or 'score'")
    instructions: str
    criteria: Optional[Any] = None

class SystemOneRequest(BaseModel):
    model: Optional[str] = "QJev-1.5B"
    state: Any
    questions: Dict[str, SystemOneQuestion]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "QJev-1.5B"
    messages: List[ChatMessage]
    choices: Optional[List[str]] = None
    temperature: Optional[float] = 1.0

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model_instance is not None}

@app.post("/v1/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model is loading or not initialized")
    try:
        res = model_instance.decide(
            context=req.context,
            choices=req.choices,
            question=req.question,
            temperature=req.temperature or 1.0
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """
    Official System One wire format for JevBench and BenchmarkHeaven.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model is loading or not initialized")
    
    answers = {}
    state_str = str(req.state)
    
    for key, q in req.questions.items():
        q_type = q.type.lower()
        instructions = q.instructions
        
        if q_type == "noul":
            res = model_instance.decide(
                context=state_str,
                choices=["false", "true"],
                question=instructions,
                temperature=1.0
            )
            p_true = float(res["probabilities"].get("true", 50.0)) / 100.0
            answers[key] = {"type": "noul", "noul": p_true}
            
        elif q_type == "choice":
            criteria = q.criteria or {}
            if isinstance(criteria, dict):
                opt_keys = list(criteria.keys())
                choices = [f"{k}: {criteria[k]}" for k in opt_keys]
                res = model_instance.decide(context=state_str, choices=choices, question=instructions)
                prob_dict = {k: float(res["probabilities"][c]) / 100.0 for k, c in zip(opt_keys, choices)}
                total = sum(prob_dict.values())
                if total > 0:
                    prob_dict = {k: v / total for k, v in prob_dict.items()}
                answers[key] = {"type": "choice", "probabilities": prob_dict}
            else:
                answers[key] = {"type": "choice", "probabilities": {}}
                
        elif q_type == "score":
            criteria = q.criteria or []
            choices = [str(c) for c in criteria]
            res = model_instance.decide(context=state_str, choices=choices, question=instructions)
            prob_dict = {str(i): float(res["probabilities"][c]) / 100.0 for i, c in enumerate(choices)}
            total = sum(prob_dict.values())
            if total > 0:
                prob_dict = {k: v / total for k, v in prob_dict.items()}
            answers[key] = {"type": "score", "probabilities": prob_dict}
            
    return {"answers": answers}

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible endpoint for drop-in usage with langchain, litellm, autogen, etc.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model is loading or not initialized")
    
    # Extract last user message as context
    user_content = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    choices = req.choices or ["Approve", "Reject", "Escalate"]
    
    res = model_instance.decide(context=user_content, choices=choices, temperature=req.temperature or 1.0)
    
    content_str = f"Best Decision: {res['best_choice']} (Confidence: {res['confidence']}%)\nDistribution: {res['probabilities']}"
    
    return {
        "id": "chatcmpl-qjev-1.5b",
        "object": "chat.completion",
        "created": 1700000000,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_str
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(user_content.split()),
            "completion_tokens": 1,
            "total_tokens": len(user_content.split()) + 1
        }
    }

def start_server():
    parser = argparse.ArgumentParser(description="QJev Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--model-path", default="export/QJev-1.5B-Phase3", help="Model path")
    parser.add_argument("--device", default=None, help="Device (cuda / cpu)")
    args = parser.parse_args()
    
    global model_instance
    print(f"Loading QJev model from {args.model_path}...")
    model_instance = QJev(model_id_or_path=args.model_path, device=args.device)
    
    print(f"\n🚀 QJev Server running on http://{args.host}:{args.port}")
    print(f"  • REST API:           POST http://{args.host}:{args.port}/v1/decide")
    print(f"  • System One Protocol: POST http://{args.host}:{args.port}/v1/systemone")
    print(f"  • OpenAI Compatible:  POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  • Health Check:       GET  http://{args.host}:{args.port}/health\n")
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    start_server()
