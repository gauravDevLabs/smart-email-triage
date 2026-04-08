import os
import sys

# Ensure project root is on the path when run as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import AgentAction, EmailCategory, EmailPriority
from environment import SmartEmailTriageEnv

app = FastAPI(title="SmartEmailTriageEnv", version="1.0.0")

# Single shared environment instance (one session per Space)
env = SmartEmailTriageEnv()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()


class ActionRequest(BaseModel):
    category: EmailCategory
    priority: EmailPriority
    should_archive: bool
    response_draft: Optional[str] = None
    reasoning: Optional[str] = None


@app.post("/step")
def step(action_req: ActionRequest):
    if env.is_done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")
    action = AgentAction(**action_req.model_dump())
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump() if obs else None,
        "reward": reward,
        "done": done,
        "info": {
            "email_id": info["email_id"],
            "is_correct": info["is_correct"],
        },
    }


@app.get("/state")
def state():
    return env.state().model_dump()


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
