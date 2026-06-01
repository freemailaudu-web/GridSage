import importlib.metadata
import math
import sys
import uuid
from typing import List

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from .agent import chat_with_agent
from .experiment_logger import (
    export_records,
    log_chat_record,
    log_run_start,
    recent_chat_records,
    recent_run_records,
)
from .executor import run_lvgs_simulation
from .grid_snapshot import build_grid_snapshot
from .explicit_command_parser import merge_explicit_global_commands
from .schema import ChatRequest, ChatResponse, DeltaCommand, ScenarioConfig, SkillMatchInfo, ValidationMessage
from .skill_prompt_builder import build_skill_context
from .skill_registry import skill_registry
from .skill_retriever import retrieve_skills
from .skill_validator import validate_with_skills, validation_status
from .validation import validate_scenario

app = FastAPI(title="LVGS Backend API", description="Scenario-Skill-Augmented LVGS Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value

sessions = {}
task_states = {}


def _dump_model(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _as_commands(commands: List[DeltaCommand]) -> List[DeltaCommand]:
    return [cmd if isinstance(cmd, DeltaCommand) else DeltaCommand(**cmd) for cmd in commands]


def _normalize_evaluation_targets_value(value):
    targets = []
    if not isinstance(value, list):
        return value
    for item in value:
        if isinstance(item, dict):
            algo = str(item.get("algo") or item.get("algo_name") or "").strip()
            if not algo:
                continue
            normalized = dict(item)
            normalized["algo"] = "Baseline" if algo.lower() == "baseline" else algo.upper()
            normalized.pop("algo_name", None)
            targets.append(normalized)
        else:
            algo = str(item or "").strip()
            if not algo:
                continue
            targets.append({"algo": "Baseline" if algo.lower() == "baseline" else algo.upper()})
    return targets


def _normalize_disabled_devices_value(value):
    normalized = {}
    if not isinstance(value, dict):
        return normalized
    aliases = {
        "gen": "generator",
        "gens": "generator",
        "generators": "generator",
        "发电机": "generator",
        "pvwind": "pv",
        "光伏": "pv",
        "wind": "wind",
        "风电": "wind",
        "ess": "ess",
        "储能": "ess",
        "ev": "ev_station",
        "evstation": "ev_station",
        "charging_station": "ev_station",
        "充电站": "ev_station",
        "sop": "sop",
        "nop": "nop",
    }
    for node, devices in value.items():
        node_id = str(node or "").strip()
        if not node_id:
            continue
        if node_id.lower().startswith("b"):
            node_id = "b" + node_id[1:]
        elif node_id.isdigit():
            node_id = f"b{node_id}"
        normalized.setdefault(node_id, {})
        if not isinstance(devices, dict):
            continue
        for device_type, ids in devices.items():
            key = aliases.get(str(device_type or "").strip().lower(), str(device_type or "").strip().lower())
            if not key:
                continue
            if ids is True or ids is None:
                id_list = ["*"]
            elif isinstance(ids, (list, tuple, set)):
                id_list = [str(item).strip() for item in ids if str(item).strip()]
            else:
                id_list = [str(ids).strip()] if str(ids).strip() else ["*"]
            normalized[node_id].setdefault(key, [])
            for item in id_list:
                if item not in normalized[node_id][key]:
                    normalized[node_id][key].append(item)
    return normalized


def _merge_disabled_devices(existing, updates):
    merged = _normalize_disabled_devices_value(existing)
    for node, devices in _normalize_disabled_devices_value(updates).items():
        merged.setdefault(node, {})
        for device_type, ids in devices.items():
            merged[node].setdefault(device_type, [])
            for item in ids:
                if item not in merged[node][device_type]:
                    merged[node][device_type].append(item)
    return merged


def apply_delta(current_state: ScenarioConfig, commands: List[DeltaCommand]) -> ScenarioConfig:
    state_dict = _dump_model(current_state)
    commands = _as_commands(commands)
    command_targets = {cmd.target for cmd in commands}
    has_evaluation_targets = "evaluation_targets" in command_targets
    single_model_targets = {"algo_name", "target_model_steps", "specific_model_name"}
    is_single_model_request = bool(single_model_targets & command_targets) and not has_evaluation_targets

    for cmd in commands:
        if cmd.action in {
            "update_config",
            "set_rl_algo",
            "configure_disturbance",
            "set_execution_mode",
            "set_specific_model",
        }:
            if cmd.target == "rl_hyperparams" and isinstance(cmd.value, dict):
                state_dict.setdefault("rl_hyperparams", {}).update(cmd.value)
            elif cmd.target in state_dict:
                if cmd.target == "evaluation_targets":
                    state_dict[cmd.target] = _normalize_evaluation_targets_value(cmd.value)
                elif cmd.target == "disabled_devices":
                    state_dict[cmd.target] = _merge_disabled_devices(state_dict.get(cmd.target), cmd.value)
                else:
                    state_dict[cmd.target] = cmd.value

        elif cmd.action == "modify_node":
            state_dict.setdefault("node_overrides", {})
            state_dict["node_overrides"].setdefault(cmd.target, {})
            if isinstance(cmd.value, dict):
                state_dict["node_overrides"][cmd.target].update(cmd.value)

    execution_mode = str(state_dict.get("execution_mode", "")).strip().lower()
    algo_name = str(state_dict.get("algo_name", "")).strip()

    if has_evaluation_targets:
        state_dict["execution_mode"] = "evaluate"
        execution_mode = "evaluate"

    if execution_mode == "train":
        state_dict["execution_mode"] = "train"
        state_dict["specific_model_name"] = ""
        state_dict["target_model_steps"] = None
        state_dict["evaluation_targets"] = []

    if algo_name.lower() == "baseline" and not has_evaluation_targets:
        state_dict["algo_name"] = "Baseline"
        state_dict["execution_mode"] = "evaluate"
        state_dict["rl_hyperparams"] = {}
        state_dict["specific_model_name"] = ""
        state_dict["target_model_steps"] = None
        state_dict["evaluation_targets"] = []

    if is_single_model_request:
        state_dict["evaluation_targets"] = []

    return ScenarioConfig(**state_dict)


def _merge_skill_state(proposed_state: ScenarioConfig, matched_skills) -> ScenarioConfig:
    if not matched_skills:
        return proposed_state
    state_dict = _dump_model(proposed_state)
    active_ids = [match.skill_id for match in matched_skills]
    existing_ids = state_dict.get("active_skill_ids") or []
    state_dict["active_skill_ids"] = list(dict.fromkeys(existing_ids + active_ids))
    primary = matched_skills[0].skill
    if primary:
        state_dict["scenario_name"] = state_dict.get("scenario_name") or primary.name_cn
        state_dict["scenario_description"] = state_dict.get("scenario_description") or primary.description_cn
    return ScenarioConfig(**state_dict)


def _build_summary(before: ScenarioConfig, after: ScenarioConfig) -> dict:
    before_dict = _dump_model(before)
    after_dict = _dump_model(after)
    changed_fields = [
        key
        for key, value in after_dict.items()
        if key != "node_overrides" and before_dict.get(key) != value
    ]
    node_changes = {
        node: params
        for node, params in after.node_overrides.items()
        if before.node_overrides.get(node) != params
    }
    disabled_device_changes = {
        node: params
        for node, params in after.disabled_devices.items()
        if before.disabled_devices.get(node) != params
    }
    return {
        "scenario_name": after.scenario_name,
        "changed_fields": changed_fields,
        "node_changes": node_changes,
        "disabled_device_changes": disabled_device_changes,
    }


def _record_chat(request: ChatRequest, response: ChatResponse, final_state: ScenarioConfig) -> None:
    log_chat_record(
        session_id=request.session_id,
        user_input=request.user_input,
        matched_skills=response.matched_skills,
        skill_warnings=response.skill_warnings,
        delta_commands=response.delta_commands,
        validation_messages=response.validation_messages,
        proposed_state_summary=response.proposed_state_summary,
        final_state=final_state,
        api_config=request.api_config,
        status=response.status,
        error_msg=response.error_msg,
    )


@app.post("/api/session/new")
async def create_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {"state": ScenarioConfig(), "history": []}
    task_states[sid] = {"status": "idle", "logs": [], "result": {}}
    return {"session_id": sid, "initial_state": sessions[sid]["state"]}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    sid = request.session_id
    if sid not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if task_states[sid]["status"] == "running":
        response = ChatResponse(
            thoughts="后台任务正在运行，当前会话暂时锁定，请等待仿真或训练结束。",
            status="error",
            error_msg="Conversation locked during execution",
        )
        _record_chat(request, response, sessions[sid]["state"])
        return response

    session = sessions[sid]
    current_state = session["state"]

    matched_skills, skill_warnings = retrieve_skills(request.user_input, current_state)
    skill_context = build_skill_context(matched_skills, current_state)

    response = await chat_with_agent(
        user_input=request.user_input,
        current_state=current_state,
        api_config=request.api_config,
        memory=session["history"][-6:],
        matched_skills=matched_skills,
        skill_context=skill_context,
    )
    response.matched_skills = [SkillMatchInfo(**match.public_dict()) for match in matched_skills]
    response.skill_warnings = skill_warnings
    explicit_commands = merge_explicit_global_commands(request.user_input, response.delta_commands)
    if response.status != "success" and explicit_commands:
        response.status = "success"
        response.error_msg = None
        response.thoughts = "已识别到明确的配置修改请求，正在更新 ScenarioConfig。"
    if response.status == "success":
        response.delta_commands = _as_commands(explicit_commands)

    if response.status == "success" and response.delta_commands:
        try:
            proposed_state = apply_delta(current_state, response.delta_commands)
            proposed_state = _merge_skill_state(proposed_state, matched_skills)

            is_valid, err_msg = validate_scenario(proposed_state)
            validation_messages: List[ValidationMessage] = []
            if not is_valid:
                validation_messages.append(ValidationMessage(level="reject", message=err_msg))
                response.delta_commands = []
                response.validation_messages = validation_messages
                response.status = "error"
                response.error_msg = err_msg
                response.thoughts += f"\n\n配置被通用安全校验拒绝：{err_msg}"
                _record_chat(request, response, current_state)
                return response

            validation_messages.extend(validate_with_skills(proposed_state, matched_skills))
            status = validation_status(validation_messages)
            response.validation_messages = validation_messages

            if status == "reject":
                response.delta_commands = []
                response.status = "error"
                response.error_msg = "; ".join(msg.message for msg in validation_messages if msg.level == "reject")
                response.thoughts += "\n\n配置被 Scenario Skill 校验拒绝：\n" + "\n".join(
                    f"- {msg.message}" for msg in validation_messages if msg.level == "reject"
                )
                _record_chat(request, response, current_state)
                return response

            warnings = [msg.message for msg in validation_messages if msg.level == "warning"]
            proposed_dict = _dump_model(proposed_state)
            proposed_dict["validation_warnings"] = warnings
            proposed_state = ScenarioConfig(**proposed_dict)

            session["state"] = proposed_state
            response.proposed_state_summary = _build_summary(current_state, proposed_state)
            if warnings:
                response.thoughts += "\n\nScenario Skill 提醒：\n" + "\n".join(f"- {item}" for item in warnings)

            session["history"].append({"role": "user", "content": request.user_input})
            session["history"].append({"role": "assistant", "content": response.thoughts})
        except Exception as exc:
            response.status = "error"
            response.error_msg = str(exc)
            response.thoughts += f"\n\n平台合并或校验 Delta Commands 失败：{exc}"
    elif response.status == "success":
        session["history"].append({"role": "user", "content": request.user_input})
        session["history"].append({"role": "assistant", "content": response.thoughts})

    _record_chat(request, response, session["state"])
    return response


@app.get("/api/state/{session_id}")
async def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]["state"]


@app.get("/api/grid_snapshot/{session_id}")
async def get_grid_snapshot(session_id: str):
    """Return the full grid topology + per-node device list + user modifications.

    This endpoint does NOT start any simulation or training. It only reads
    the current session ScenarioConfig and the grid parameter Excel file to
    build a read-only snapshot for the frontend right-side panel.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state: ScenarioConfig = sessions[session_id]["state"]
    snapshot = build_grid_snapshot(state)
    if hasattr(snapshot, "model_dump"):
        return snapshot.model_dump()
    return snapshot.dict()


@app.get("/api/health")
async def health():
    packages = {}
    for name in [
        "openpyxl",
        "matplotlib",
        "gymnasium",
        "feasytools",
        "pandas",
        "pyomo",
        "gurobipy",
        "torch",
        "stable_baselines3",
        "tensorboard",
        "rich",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "status": "ok",
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": packages,
    }


@app.get("/api/skills")
async def list_skills():
    return {"skills": skill_registry.summaries()}


@app.get("/api/skills/{skill_id}")
async def get_skill(skill_id: str):
    skill = skill_registry.get(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill


@app.post("/api/skills/match")
async def match_skills(payload: dict):
    session_id = payload.get("session_id")
    user_input = payload.get("user_input", "")
    state = sessions.get(session_id, {}).get("state", ScenarioConfig())
    matches, warnings = retrieve_skills(user_input, state)
    return {
        "matched_skills": [match.public_dict() for match in matches],
        "warnings": warnings,
    }


@app.post("/api/run_task")
async def run_task(session_id: str, background_tasks: BackgroundTasks):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if task_states[session_id]["status"] == "running":
        return {"status": "already_running"}

    task_states[session_id]["status"] = "running"
    task_states[session_id]["logs"] = ["准备启动 VGridSim 评估引擎..."]
    task_states[session_id]["timestamp_start"] = log_run_start(session_id, sessions[session_id]["state"])
    background_tasks.add_task(run_lvgs_simulation, session_id, sessions[session_id]["state"], task_states)
    return {"status": "started"}


@app.get("/api/task_status/{session_id}")
async def get_task_status(session_id: str):
    if session_id not in task_states:
        raise HTTPException(status_code=404, detail="Task not found")
    return _json_safe(task_states[session_id])


@app.get("/api/experiments/chat_records")
async def get_chat_records(limit: int = Query(100, ge=1, le=10000)):
    return {"records": recent_chat_records(limit=limit)}


@app.get("/api/experiments/run_records")
async def get_run_records(limit: int = Query(100, ge=1, le=10000)):
    return {"records": recent_run_records(limit=limit)}


@app.get("/api/experiments/session/{session_id}")
async def get_session_records(session_id: str, limit: int = Query(1000, ge=1, le=10000)):
    return {
        "chat_records": recent_chat_records(limit=limit, session_id=session_id),
        "run_records": recent_run_records(limit=limit, session_id=session_id),
    }


@app.get("/api/experiments/export")
async def export_experiment_records(kind: str = "jsonl", session_id: str = None):
    media_type = "text/csv" if kind.lower() == "csv" else "application/x-ndjson"
    return PlainTextResponse(export_records(kind=kind, session_id=session_id), media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
