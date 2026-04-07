from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import ENV_NAME, ENV_VERSION, EcoCloudEnv
from env.models import Action, EnvState, Observation, StepResult, TaskSummary


class ResetRequest(BaseModel):
    task_id: str


class ResetResponse(BaseModel):
    observation: Observation
    task: TaskSummary
    max_steps: int


app = FastAPI(title=ENV_NAME, version=ENV_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EcoCloudEnv()


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": ENV_NAME,
        "version": ENV_VERSION,
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "tasks": "/tasks",
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "validate": "/validate",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks", response_model=list[TaskSummary])
def tasks() -> list[TaskSummary]:
    return [TaskSummary(**task) for task in env.list_tasks()]


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest) -> ResetResponse:
    try:
        observation = env.reset(request.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    summary = next(task for task in env.list_tasks() if task["task_id"] == request.task_id)
    return ResetResponse(
        observation=observation,
        task=TaskSummary(**summary),
        max_steps=summary["max_steps"],
    )


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    return env.step(action)


@app.get("/state", response_model=EnvState)
def state() -> EnvState:
    return env.state()


@app.get("/grade/{task_id}")
def grade(task_id: str) -> dict[str, object]:
    if task_id != env.task_id:
        raise HTTPException(status_code=400, detail="Grade is only available for the active task.")
    return env.grade_current_task()


@app.get("/validate")
def validate() -> dict[str, object]:
    return env.validate_spec()
