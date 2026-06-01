import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LOG_DIR = Path(__file__).parent / "logs"
CHAT_LOG = LOG_DIR / "chat_records.jsonl"
RUN_LOG = LOG_DIR / "run_records.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, list):
        return [dump_model(item) for item in value]
    if isinstance(value, dict):
        return {key: dump_model(item) for key, item in value.items()}
    return value


def sanitize_api_config(api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    api_config = api_config or {}
    return {
        "model_name": api_config.get("model_name", ""),
        "base_url": api_config.get("base_url", ""),
    }


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path, limit: int = 100, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if session_id and record.get("session_id") != session_id:
                continue
            rows.append(record)
    return rows[-limit:]


def derive_validation_status(messages: Iterable[Any], response_status: str = "success") -> str:
    levels = {getattr(item, "level", None) or item.get("level") for item in messages or []}
    if "reject" in levels:
        return "reject"
    if response_status == "error":
        return "error"
    if "warning" in levels:
        return "warning"
    return "pass"


def log_chat_record(
    *,
    session_id: str,
    user_input: str,
    matched_skills: Iterable[Any],
    skill_warnings: Iterable[str],
    delta_commands: Iterable[Any],
    validation_messages: Iterable[Any],
    proposed_state_summary: Dict[str, Any],
    final_state: Any,
    api_config: Optional[Dict[str, Any]],
    status: str,
    error_msg: Optional[str] = None,
) -> None:
    matched = list(matched_skills or [])
    matched_ids = [getattr(item, "skill_id", None) or item.get("skill_id") for item in matched]
    record = {
        "record_type": "chat",
        "timestamp": utc_now_iso(),
        "session_id": session_id,
        "user_input": user_input,
        "matched_skill_ids": [item for item in matched_ids if item],
        "primary_skill_id": next((item for item in matched_ids if item), None),
        "skill_warnings": list(skill_warnings or []),
        "delta_commands": dump_model(list(delta_commands or [])),
        "validation_status": derive_validation_status(validation_messages, status),
        "validation_messages": dump_model(list(validation_messages or [])),
        "proposed_state_summary": dump_model(proposed_state_summary or {}),
        "final_state": dump_model(final_state),
        "llm_model_name": (api_config or {}).get("model_name", ""),
        "api_base_url": (api_config or {}).get("base_url", ""),
        "status": "success" if status == "success" else "error",
        "error_msg": error_msg,
    }
    _append_jsonl(CHAT_LOG, record)


def log_run_start(session_id: str, state: Any) -> str:
    timestamp = utc_now_iso()
    record = {
        "record_type": "run_task_start",
        "timestamp_start": timestamp,
        "session_id": session_id,
        "active_skill_ids": list(getattr(state, "active_skill_ids", []) or []),
        "scenario_name": getattr(state, "scenario_name", ""),
        "scenario_config_snapshot": dump_model(state),
        "execution_mode": getattr(state, "execution_mode", ""),
        "algo_name": getattr(state, "algo_name", ""),
        "total_timesteps": getattr(state, "rl_hyperparams", {}).get("total_timesteps"),
        "evaluation_targets": dump_model(getattr(state, "evaluation_targets", [])),
        "status": "started",
    }
    _append_jsonl(RUN_LOG, record)
    return timestamp


def log_run_finish(
    *,
    session_id: str,
    state: Any,
    timestamp_start: Optional[str],
    task_state: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    result = task_state.get("result", {}) or {}
    all_metrics = metrics or result.get("all_metrics") or {}
    record = {
        "record_type": "run_task",
        "timestamp_start": timestamp_start,
        "timestamp_end": utc_now_iso(),
        "session_id": session_id,
        "active_skill_ids": list(getattr(state, "active_skill_ids", []) or []),
        "scenario_name": getattr(state, "scenario_name", ""),
        "scenario_config_snapshot": dump_model(state),
        "execution_mode": getattr(state, "execution_mode", ""),
        "algo_name": getattr(state, "algo_name", ""),
        "total_timesteps": getattr(state, "rl_hyperparams", {}).get("total_timesteps"),
        "evaluation_targets": dump_model(getattr(state, "evaluation_targets", [])),
        "status": task_state.get("status", "error"),
        "metrics": dump_model(all_metrics if not all_metrics.get("is_comparison") else {}),
        "comparison_metrics": dump_model(result.get("comparison_metrics") or all_metrics.get("comparison_metrics") or {}),
        "error_message": error_message,
        "logs_tail": list(task_state.get("logs", [])[-30:]),
    }
    _append_jsonl(RUN_LOG, record)


def recent_chat_records(limit: int = 100, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return _read_jsonl(CHAT_LOG, limit=limit, session_id=session_id)


def recent_run_records(limit: int = 100, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return _read_jsonl(RUN_LOG, limit=limit, session_id=session_id)


def export_records(kind: str = "jsonl", session_id: Optional[str] = None) -> str:
    records = recent_chat_records(limit=100000, session_id=session_id) + recent_run_records(
        limit=100000, session_id=session_id
    )
    records.sort(key=lambda item: item.get("timestamp") or item.get("timestamp_start") or "")
    if kind.lower() == "csv":
        fields = sorted({key for record in records for key in record.keys()})
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(record.get(key), ensure_ascii=False) if isinstance(record.get(key), (dict, list)) else record.get(key) for key in fields})
        return buf.getvalue()
    return "\n".join(json.dumps(record, ensure_ascii=False, default=str) for record in records)
