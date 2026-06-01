import sys
import os
import io
import asyncio
import json
import re
from contextlib import redirect_stdout, redirect_stderr

# 必须用 insert(0) 确保 vgridsim_core 优先级最高，避免被系统其他同名模块遥体
_CORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgridsim_core")
if _CORE_PATH not in sys.path:
    sys.path.insert(0, _CORE_PATH)

import config
from scenario_fingerprint import build_scenario_fingerprint, build_scenario_signature
from .schema import ScenarioConfig
from .experiment_logger import log_run_finish
from .result_interpreter import interpret_results_with_skills


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _prepare_training_subprocess_env(train_steps) -> dict:
    env_copy = os.environ.copy()
    env_copy["TRAINING_TOTAL_TIMESTEPS"] = str(train_steps)
    env_copy["PYTHONUTF8"] = "1"
    env_copy["PYTHONIOENCODING"] = "utf-8"
    env_copy["TQDM_DISABLE"] = "1"
    env_copy["LVGS_SUPPRESS_WINDOWS_ERROR_DIALOG"] = "1"
    env_copy["LVGS_FAST_TRAINING"] = "1"
    return env_copy


def _clean_process_log_line(line: str) -> str:
    text = _ANSI_RE.sub("", str(line or "")).strip()
    return text.replace("\ufffd", "")


def _extract_steps(value):
    """从 int/字符串中提取训练步数，例如 1000、"1000步"、"sac_2stage_1000steps"。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value)
    m = re.search(r"(\d+)\s*(?:steps?|timesteps?|步)?", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _read_model_manifest(item_path: str) -> dict:
    manifest_path = os.path.join(item_path, "model_manifest.json") if os.path.isdir(item_path) else ""
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _pick_zip_from_item(item_path: str):
    """给定 models 目录下的文件/文件夹，返回实际可加载的模型 zip。"""
    if os.path.isfile(item_path) and item_path.lower().endswith(".zip"):
        return item_path
    if not os.path.isdir(item_path):
        return None

    # 新版本目录优先加载训练结束时的最终模型；旧目录再兼容 best_model.zip。
    preferred = ["final_model.zip", "best_model.zip"]
    for filename in preferred:
        cand = os.path.join(item_path, filename)
        if os.path.exists(cand):
            return cand

    sub_files = [os.path.join(item_path, f) for f in os.listdir(item_path) if f.lower().endswith(".zip")]
    if not sub_files:
        return None
    sub_files.sort(key=os.path.getmtime, reverse=True)
    return sub_files[0]


def _find_requested_model(
    models_dir: str,
    algo: str,
    target_steps=None,
    specific_model_name: str = "",
    scenario_fingerprint: str = "",
    require_scenario_match: bool = False,
):
    """
    查找模型时必须尊重用户指定的训练步数。
    - 若指定了 target_steps 但找不到，直接报错，不再静默退回“最新模型”。
    - 若未指定步数/名称，则返回该算法最新模型。
    """
    if not os.path.exists(models_dir):
        return None

    algo_key = str(algo).upper()
    algo_lower = str(algo).lower()
    specific = (specific_model_name or "").strip().lower()
    target_steps = _extract_steps(target_steps) or _extract_steps(specific)

    all_items = [os.path.join(models_dir, item) for item in os.listdir(models_dir)]
    all_items.sort(key=os.path.getmtime, reverse=True)

    matches = []
    for item_path in all_items:
        model_zip = _pick_zip_from_item(item_path)
        if not model_zip:
            continue

        item_name = os.path.basename(item_path).lower()
        zip_name = os.path.basename(model_zip).lower()
        manifest = _read_model_manifest(item_path)
        manifest_text = json.dumps(manifest, ensure_ascii=False).lower() if manifest else ""

        manifest_algo = str(manifest.get("algo", "")).upper()
        algo_match = (manifest_algo == algo_key) or (algo_lower in item_name) or (algo_lower in zip_name)
        if not algo_match:
            continue

        if target_steps is not None:
            manifest_steps = _extract_steps(manifest.get("total_timesteps"))
            name_steps = _extract_steps(item_name) or _extract_steps(zip_name)
            if manifest_steps != target_steps and name_steps != target_steps:
                continue
        elif specific:
            if specific not in item_name and specific not in zip_name and specific not in manifest_text:
                continue

        if require_scenario_match:
            manifest_fingerprint = str(manifest.get("scenario_fingerprint", "") or "")
            if not manifest_fingerprint or manifest_fingerprint != scenario_fingerprint:
                continue

        matches.append(model_zip)

    if target_steps is not None and not matches and require_scenario_match:
        return None

    if target_steps is not None and not matches:
        raise FileNotFoundError(
            f"用户指定了 {algo_key} 的 {target_steps} 步模型，但 models 目录中没有找到带有该步数版本标记的模型。"
            f"请先训练 {target_steps} 步，或确认模型目录名/manifest 中包含 {target_steps}steps。"
        )

    return matches[0] if matches else None


def _find_recent_trained_model_after(models_dir: str, algo: str, target_steps, started_at: float):
    if not os.path.exists(models_dir):
        return None

    algo_lower = str(algo).lower()
    target_steps = _extract_steps(target_steps)
    candidates = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.getmtime(item_path) < started_at:
            continue
        manifest = _read_model_manifest(item_path)
        manifest_steps = _extract_steps(manifest.get("total_timesteps"))
        item_steps = _extract_steps(item)
        manifest_algo = str(manifest.get("algo", "")).lower()
        if manifest_algo and manifest_algo != algo_lower:
            continue
        if algo_lower not in item.lower() and manifest_algo != algo_lower:
            continue
        if target_steps is not None and manifest_steps != target_steps and item_steps != target_steps:
            continue
        model_zip = _pick_zip_from_item(item_path)
        if model_zip:
            candidates.append(model_zip)

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def _state_to_scenario_settings(state: ScenarioConfig) -> dict:
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return state.dict()


def _assert_model_matches_current_scenario(model_path: str, state: ScenarioConfig, algo: str, label: str) -> dict:
    manifest = _read_model_manifest(os.path.dirname(model_path)) if model_path else {}
    current_settings = _state_to_scenario_settings(state)
    current_fingerprint = build_scenario_fingerprint(current_settings)
    model_fingerprint = str(manifest.get("scenario_fingerprint", "") or "")

    if not model_fingerprint:
        raise ValueError(
            f"已找到 {algo} 模型 {os.path.basename(model_path)}，但该模型没有场景指纹，"
            f"无法确认它是否是在当前配电网设备配置下训练得到的。为避免把旧场景模型套用到新场景，"
            f"本次评估已停止。请先在当前场景下重新训练 {algo}，再执行评估。"
        )

    if model_fingerprint != current_fingerprint:
        trained_signature = manifest.get("scenario_signature") or {}
        current_signature = build_scenario_signature(current_settings)
        changed_fields = [
            field for field in sorted(current_signature.keys())
            if trained_signature.get(field) != current_signature.get(field)
        ]
        changed_text = ", ".join(changed_fields) if changed_fields else "unknown scenario fields"
        raise ValueError(
            f"{label} 的模型场景指纹与当前场景不一致，已阻止评估。"
            f"不一致字段: {changed_text}。"
            f"请先用当前配电网设备/倍率/节点扰动重新训练 {algo}，再评估。"
        )

    return manifest


def _format_metric_cell(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _has_ev_demand(state: ScenarioConfig) -> bool:
    if float(getattr(state, "global_ev_multiplier", 1.0) or 0.0) > 0:
        return True
    for params in getattr(state, "node_overrides", {}).values():
        if float(params.get("add_ev_spots", 0) or 0) > 0:
            return True
    return False


def _filter_metrics_for_state(metrics: dict, state: ScenarioConfig) -> dict:
    if _has_ev_demand(state):
        return metrics
    return {key: value for key, value in metrics.items() if "EV充电满足率" not in key}


def _write_gui_settings_for_state(state: ScenarioConfig) -> None:
    settings_path = config.PATHS.get("gui_settings")
    settings = {}
    if settings_path and os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception:
            settings = {}

    settings.update({
        "grid_model": state.grid_model,
        "solver": state.solver,
        "start_hour": int(state.start_hour),
        "end_hour": int(state.end_hour),
        "step_minutes": int(state.step_minutes),
        "use_pv": bool(state.use_pv),
        "use_wind": bool(state.use_wind),
        "use_ess": bool(state.use_ess),
        "use_sop": bool(state.use_sop),
        "use_nop": bool(state.use_nop),
        "reconfiguration_mode": state.reconfiguration_mode,
        "selected_reconfiguration_plan_id": state.selected_reconfiguration_plan_id,
        "available_reconfiguration_plans": getattr(state, "available_reconfiguration_plans", []) or [],
        "reconfiguration_constraints": getattr(state, "reconfiguration_constraints", {}) or {},
        "reward_mode": state.reward_mode,
        "global_pv_multiplier": float(state.global_pv_multiplier),
        "global_load_multiplier": float(state.global_load_multiplier),
        "global_ev_multiplier": float(state.global_ev_multiplier),
        "time_profiles": getattr(state, "time_profiles", {}) or {},
        "node_overrides": getattr(state, "node_overrides", {}) or {},
        "disabled_devices": getattr(state, "disabled_devices", {}) or {},
    })

    if state.rl_hyperparams:
        settings.setdefault("rl_common", {})
        for key in ["learning_rate", "batch_size", "gamma"]:
            if key in state.rl_hyperparams:
                settings["rl_common"][key] = state.rl_hyperparams[key]
        algo_key = str(state.algo_name or "").upper()
        if algo_key:
            settings.setdefault("rl_specific", {}).setdefault(algo_key, {})
            for key, value in state.rl_hyperparams.items():
                if key not in {"learning_rate", "batch_size", "gamma", "total_timesteps"}:
                    settings["rl_specific"][algo_key][key] = value

    if settings_path:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _hour_from_time(t) -> int:
    value = _as_float(t, 0.0) or 0.0
    if value > 24:
        value = value / 3600.0
    return int(value) % 24


def _profile_lookup(profile: dict, key: str, hour: int, default: float) -> float:
    values = (profile or {}).get(key) or {}
    if isinstance(values, list):
        return _as_float(values[hour], default) if hour < len(values) else default
    if isinstance(values, dict):
        for candidate in (hour, str(hour), f"{hour:02d}"):
            if candidate in values:
                return _as_float(values[candidate], default)
        for range_key, value in values.items():
            text = str(range_key)
            if "-" in text:
                start, end = text.split("-", 1)
                start_hour = int(_as_float(start, -1))
                end_hour = int(_as_float(end, -1))
                if start_hour <= hour <= end_hour:
                    return _as_float(value, default)
    return default


def _profile_average(profile: dict, key: str, default: float) -> float:
    values = []
    for hour in range(24):
        values.append(_profile_lookup(profile, key, hour, default))
    return sum(values) / len(values) if values else default


def _apply_time_or_global_profiles(env, state: ScenarioConfig) -> str:
    profile = getattr(state, "time_profiles", None) or {}
    has_profile = bool(
        profile.get("load_multiplier_by_hour")
        or profile.get("pv_multiplier_by_hour")
        or profile.get("ev_multiplier_by_hour")
    )
    load_default = float(state.global_load_multiplier)
    pv_default = float(state.global_pv_multiplier)
    ev_default = float(state.global_ev_multiplier)

    if has_profile:
        return (
            f"time_profiles active: {profile.get('profile_name') or 'custom'} "
            f"(avg load x{_profile_average(profile, 'load_multiplier_by_hour', load_default):.2f}, "
            f"avg pv x{_profile_average(profile, 'pv_multiplier_by_hour', pv_default):.2f}, "
            f"avg ev x{_profile_average(profile, 'ev_multiplier_by_hour', ev_default):.2f})"
        )

    if state.global_load_multiplier != 1.0 or state.global_pv_multiplier != 1.0:
        return f"global multipliers active: load x{load_default}, pv x{pv_default}"

    return "default multipliers active"


def _normalize_evaluation_targets(state: ScenarioConfig):
    """把 state.evaluation_targets 归一成 [{algo, steps, label, specific_model_name}]。"""
    raw_targets = getattr(state, "evaluation_targets", None) or []
    targets = []
    if isinstance(raw_targets, list):
        for idx, raw in enumerate(raw_targets):
            if isinstance(raw, dict):
                algo = str(raw.get("algo") or raw.get("algo_name") or "").strip()
                steps = _extract_steps(raw.get("steps") or raw.get("target_model_steps"))
                specific = str(raw.get("specific_model_name") or raw.get("model") or "").strip()
                label = str(raw.get("label") or "").strip()
            else:
                algo = str(raw or "").strip()
                steps = None
                specific = ""
                label = ""
            if not algo:
                continue
            if algo.lower() == "baseline":
                algo = "Baseline"
            if steps is None and algo.upper() != "BASELINE":
                steps = _extract_steps(getattr(state, "target_model_steps", None))
                if steps is None:
                    steps = _extract_steps((getattr(state, "rl_hyperparams", {}) or {}).get("total_timesteps"))
            if not label:
                label = algo.upper() if algo.upper() == "BASELINE" else f"{algo.upper()}-{steps}steps" if steps else f"{algo.upper()}-latest"
            targets.append({
                "algo": algo.upper(),
                "steps": steps,
                "specific_model_name": specific,
                "label": label,
            })

    # 没有批量目标时，保持原来的单模型行为。
    if not targets:
        algo = str(getattr(state, "algo_name", "Baseline") or "Baseline").upper()
        steps = _extract_steps(getattr(state, "target_model_steps", None))
        specific = str(getattr(state, "specific_model_name", "") or "").strip()
        label = algo if algo == "BASELINE" else f"{algo}-{steps}steps" if steps else f"{algo}-latest"
        targets.append({"algo": algo, "steps": steps, "specific_model_name": specific, "label": label})

    return targets


def _build_comparison_message(comparison_metrics: dict, state: ScenarioConfig = None) -> str:
    """把多模型评估结果组织成 Markdown 表格，方便前端直接显示。"""
    if not comparison_metrics:
        return "⚠️ 没有生成任何可比较的评估结果。"

    display_metrics = {
        label: _filter_metrics_for_state(metrics, state) if state is not None else metrics
        for label, metrics in comparison_metrics.items()
    }

    priority = [
        "总成本", "总目标值", "总惩罚成本", "购电成本", "发电成本", "SOP损耗成本", "ESS放电成本",
        "电压惩罚成本", "OpenDSS失败惩罚成本", "EV未满足惩罚成本",
        "功率平衡松弛惩罚", "EV未充满惩罚", "SOP容量松弛惩罚", "NOP电压松弛惩罚",
        "精确总网损(kW)", "电压合格率(%)", "最低节点电压(pu)", "最高节点电压(pu)",
        "最大电压偏差(pu)", "电压极差(pu)", "最大线路功率流(pu)", "最大线路功率流线路",
        "光伏风电消纳率(%)", "弃光弃风量(kWh)", "EV充电满足率(%)",
        "环境累计奖励(缩放后)", "环境累计奖励(未缩放)",
    ]
    all_keys = []
    for metrics in display_metrics.values():
        for key in metrics.keys():
            if key not in all_keys and not key.startswith("_"):
                all_keys.append(key)
    selected_keys = [k for k in priority if k in all_keys]
    selected_keys += [k for k in all_keys if k not in selected_keys]

    header = ["模型/算法"] + selected_keys
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for label, metrics in display_metrics.items():
        row = [label] + [_format_metric_cell(metrics.get(k, "-")) for k in selected_keys]
        lines.append("| " + " | ".join(row) + " |")

    best_line = ""
    candidates = []
    for label, metrics in display_metrics.items():
        value = metrics.get("总目标值", metrics.get("总成本"))
        if isinstance(value, (int, float)):
            candidates.append((float(value), label))
    if candidates:
        best_objective, best_label = min(candidates, key=lambda x: x[0])
        baseline_candidates = [
            value for value, label in candidates
            if str(label).strip().upper() == "BASELINE"
        ]
        if baseline_candidates and str(best_label).strip().upper() != "BASELINE":
            baseline_objective = baseline_candidates[0]
            if best_objective < baseline_objective - 1e-6:
                best_line = (
                    f"\n\n⚠️ **口径校验异常**：{best_label} 的总目标值 {best_objective:.4f} "
                    f"低于 Baseline {baseline_objective:.4f}。Baseline 是全局优化基准，"
                    f"此时不应直接判定 RL 最优，请优先检查惩罚项和目标函数口径。"
                )
            else:
                best_line = f"\n\n🏆 按 **总目标值最低** 计算，当前最优为：**{best_label}**，总目标值 {best_objective:.4f}。"
        else:
            best_line = f"\n\n🏆 按 **总目标值最低** 计算，当前最优为：**{best_label}**，总目标值 {best_objective:.4f}。"

    return "✅ **多模型/多算法评估完成！**\n\n📊 **对比结果如下：**\n\n" + "\n".join(
        lines) + best_line + "\n\n💡 所有目标使用同一场景配置与相同随机种子 seed=0 评估，便于横向比较。"


async def run_lvgs_simulation(session_id: str, state: ScenarioConfig, task_states: dict):
    """
    底层仿真调用的桥接执行器。
    负责将 Pydantic 的 ScenarioConfig 映射回 VGridSim 习惯读取的 config 全局变量，并调用环境。
    """
    task_states[session_id]["logs"].append("=> 开始构建并映射 LVGS 最新配置至底层引擎...")
    if getattr(state, "active_skill_ids", None):
        task_states[session_id]["logs"].append("=> Active Scenario Skills: " + ", ".join(state.active_skill_ids))

    # 1. 映射配置到原有的 config 结构
    try:
        def to_scalar(v):
            if isinstance(v, (list, tuple)):
                return v[0] if v else 0
            return v

        # 正确映射 CORE_PARAMS - 使用与 config.py 中完全一致的 flat key 结构
        config.CORE_PARAMS["grid_model"] = state.grid_model
        config.CORE_PARAMS["reward_mode"] = state.reward_mode
        config.CORE_PARAMS["distributed_energy"]["pv"] = state.use_pv
        config.CORE_PARAMS["distributed_energy"]["wind"] = state.use_wind
        config.CORE_PARAMS["distributed_energy"]["ess"] = state.use_ess
        config.CORE_PARAMS["sop_nodes_active"] = state.use_sop
        config.CORE_PARAMS["nop_nodes_active"] = state.use_nop
        config.CORE_PARAMS["reconfiguration_mode"] = state.reconfiguration_mode
        config.CORE_PARAMS["selected_reconfiguration_plan_id"] = state.selected_reconfiguration_plan_id
        config.CORE_PARAMS["available_reconfiguration_plans"] = getattr(state, "available_reconfiguration_plans", []) or []
        config.CORE_PARAMS["reconfiguration_constraints"] = getattr(state, "reconfiguration_constraints", {}) or {}

        # 时序与步长挂载 (强制转为标量)
        config.CORE_PARAMS["start_hour"] = int(to_scalar(state.start_hour))
        config.CORE_PARAMS["end_hour"] = int(to_scalar(state.end_hour))
        config.CORE_PARAMS["step_minutes"] = int(to_scalar(state.step_minutes))

        # [FIX 1] 求解器字段映射 - 将用户选择的 solver 传递给底层引擎
        config.CORE_PARAMS["solver"] = state.solver
        task_states[session_id]["logs"].append(f"=> 已绑定求解器: {state.solver}")

        # [FIX 3] EV 规模缩放因子映射 - 将 global_ev_multiplier 写入核心参数 (强制转为标量)
        config.CORE_PARAMS["global_load_multiplier"] = float(to_scalar(state.global_load_multiplier))
        config.CORE_PARAMS["global_pv_multiplier"] = float(to_scalar(state.global_pv_multiplier))
        config.CORE_PARAMS["time_profiles"] = getattr(state, "time_profiles", {}) or {}
        config.CORE_PARAMS["node_overrides"] = getattr(state, "node_overrides", {}) or {}
        config.CORE_PARAMS["disabled_devices"] = getattr(state, "disabled_devices", {}) or {}
        config.CORE_PARAMS["global_ev_multiplier"] = float(to_scalar(state.global_ev_multiplier))
        task_states[session_id]["logs"].append(
            f"=> 已绑定 EV 规模缩放因子: x{config.CORE_PARAMS['global_ev_multiplier']}")

        # 强化学习超参数挂载
        if state.rl_hyperparams:
            algo_key = state.algo_name.strip().upper()
            if algo_key not in config.RL_HYPERPARAMS.get("algo_specific", {}):
                config.RL_HYPERPARAMS["algo_specific"][algo_key] = {}

            for k, v in state.rl_hyperparams.items():
                if k in ["learning_rate", "batch_size", "gamma"]:
                    config.RL_HYPERPARAMS["common"][k] = v
                else:
                    config.RL_HYPERPARAMS["algo_specific"][algo_key][k] = v
            task_states[session_id]["logs"].append(f"=> 已挂载 RL 自定义超参数: {len(state.rl_hyperparams)} 项")

        task_states[session_id]["logs"].append(f"=> 已挂载节点突变数量: {len(state.node_overrides)}")
        task_states[session_id]["logs"].append(f"=> 已禁用默认设备节点数量: {len(getattr(state, 'disabled_devices', {}) or {})}")
        _write_gui_settings_for_state(state)
        task_states[session_id]["logs"].append("=> 已写入训练/评估子进程场景配置 gui_settings.json")
    except Exception as map_err:
        task_states[session_id]["logs"].append(f"=> 代理层参数映射失败: {map_err}")

    # 2. 在线程池中运行核心评估（redirect_stdout 必须在线程内，不能跨线程）
    task_states[session_id]["logs"].append(f"=> 正在分配后台计算线程并启动 {state.algo_name}...")

    try:
        loop = asyncio.get_running_loop()
        real_metrics, captured_logs = await loop.run_in_executor(
            None, _run_core_evaluation_sync, state, session_id, task_states
        )

        # 把线程内捕获的打印日志转入前端
        if captured_logs:
            task_states[session_id]["logs"].extend(captured_logs[-50:])

            # 根据指标中的标记区分纯训练模式和评估模式
            if real_metrics.get("is_train_only", False):
                task_states[session_id]["status"] = "train_completed"
                task_states[session_id]["result"] = {
                    "message": f"成功完成 {state.algo_name} 模型的训练。",
                    "status": "TrainSuccess"
                }
                task_states[session_id]["logs"].append(f"=> ✔ 训练任务结束！本次不触发仿真评估。")
            elif real_metrics.get("_error"):
                task_states[session_id]["status"] = "error"
                task_states[session_id]["result"] = {
                    "message": f"仿真未能完成：{real_metrics.get('_error')}",
                    "status": "SimulationFailed",
                    "all_metrics": real_metrics,
                }
                task_states[session_id]["logs"].append(f"=> ✘ 仿真失败：{real_metrics.get('_error')}")
            else:
                task_states[session_id]["status"] = "completed"

                if real_metrics.get("is_comparison", False):
                    comparison_metrics = real_metrics.get("comparison_metrics", {})
                    detailed_message = _build_comparison_message(comparison_metrics, state)

                    # 用第一项填充旧版前端摘要字段，完整对比数据放在 comparison_metrics/all_metrics 中。
                    first_metrics = next(iter(comparison_metrics.values()), {}) if comparison_metrics else {}
                    task_states[session_id]["result"] = {
                        "message": detailed_message,
                        "status": "ComparisonSuccess",
                        "cost": float(first_metrics.get("总成本", 0.0)) if isinstance(first_metrics.get("总成本", 0.0),
                                                                                      (int, float)) else 0.0,
                        "voltage_pass_rate": float(first_metrics.get("电压合格率(%)", 100.0)) if isinstance(
                            first_metrics.get("电压合格率(%)", 100.0), (int, float)) else 100.0,
                        "grid_loss_kw": float(first_metrics.get("精确总网损(kW)", 0.0)) if isinstance(
                            first_metrics.get("精确总网损(kW)", 0.0), (int, float)) else 0.0,
                        "comparison_metrics": comparison_metrics,
                        "all_metrics": real_metrics
                    }
                    task_states[session_id]["logs"].append(
                        f"=> ✔ 多模型评估完成！共对比 {len(comparison_metrics)} 个目标。")
                else:
                    # ================= [新增] 动态拼接所有的评估指标 =================
                    # 遍历 real_metrics 字典，把所有项转为 Markdown 列表格式（保留 4 位小数）
                    display_metrics = _filter_metrics_for_state(real_metrics, state)
                    metrics_details = "\n".join([
                        f"- **{k}**: {round(v, 4) if isinstance(v, float) else v}"
                        for k, v in display_metrics.items()
                        if k != "is_train_only"
                    ])

                    # 构建新的丰富版回复文本
                    detailed_message = (
                        f"✅ **仿真运行成功结束！**\n\n"
                        f"📊 **详细评估指标如下：**\n{metrics_details}\n\n"
                        f"💡 *您可继续使用自然语言发起下一次参数修改。*"
                    )
                    # ===============================================================

                    task_states[session_id]["result"] = {
                        "message": detailed_message,  # 将丰富的文本传给前端的 message 字段
                        "status": "Success",
                        "cost": float(real_metrics.get("总成本", 0.0)),
                        "voltage_pass_rate": float(real_metrics.get("电压合格率(%)", 100.0)),
                        "grid_loss_kw": float(real_metrics.get("精确总网损(kW)", 0.0)),
                        "all_metrics": real_metrics  # 完整数据原封不动传给前端，方便以后前端画图或做表格
                    }
                    task_states[session_id]["logs"].append(
                        f"=> ✔ 仿真运行成功！总成本: {real_metrics.get('总成本', 0.0):.2f}")

        comparison_metrics = real_metrics.get("comparison_metrics", {}) if isinstance(real_metrics, dict) else {}
        skill_interpretation = interpret_results_with_skills(
            getattr(state, "active_skill_ids", []),
            real_metrics,
            state=state,
            comparison_metrics=comparison_metrics,
            task_status=task_states[session_id].get("status", ""),
        )
        if skill_interpretation and task_states[session_id].get("result", {}).get("message"):
            task_states[session_id]["result"]["message"] += skill_interpretation
        log_run_finish(
            session_id=session_id,
            state=state,
            timestamp_start=task_states[session_id].get("timestamp_start"),
            task_state=task_states[session_id],
            metrics=real_metrics,
        )

    except FileNotFoundError as err:
        task_states[session_id]["status"] = "error"
        task_states[session_id]["logs"].append(f"❌ 关键权重文件缺失: {str(err)}")
        task_states[session_id]["logs"].append("=> 请对大模型说：'切换回 Baseline 优化器并重新运行'")
        log_run_finish(
            session_id=session_id,
            state=state,
            timestamp_start=task_states[session_id].get("timestamp_start"),
            task_state=task_states[session_id],
            error_message=str(err),
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        with open("error_dump.txt", "w", encoding="utf-8") as err_f:
            err_f.write(tb)
        task_states[session_id]["status"] = "error"
        task_states[session_id]["logs"].append(f"❌ 实验执行层发生致命崩溃: {str(e)}")
        # 把完整堆栈写入日志，方便精确定位
        tb_lines = tb.split('\n')
        task_states[session_id]["logs"].extend([f"  {l}" for l in tb_lines if l.strip()][-20:])
        log_run_finish(
            session_id=session_id,
            state=state,
            timestamp_start=task_states[session_id].get("timestamp_start"),
            task_state=task_states[session_id],
            error_message=str(e),
        )


def _run_core_evaluation_sync(state: ScenarioConfig, session_id: str = None, task_states: dict = None):
    """
    同步隔离区，运行在独立线程中。在此处捕获 stdout，返回 (metrics, log_lines) 元组。
    """
    import traceback
    import io
    from contextlib import redirect_stdout, redirect_stderr

    f = io.StringIO()

    with redirect_stdout(f), redirect_stderr(f):
        # --- 路径诊断日志 ---
        _CORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgridsim_core")
        if _CORE_PATH not in sys.path:
            sys.path.insert(0, _CORE_PATH)
        print(f"[LVGS-Engine] vgridsim_core path: {_CORE_PATH}")
        print(f"[LVGS-Engine] path_exists: {os.path.exists(_CORE_PATH)}")

        # --- 验证 Excel 文件 ---
        from config import PATHS
        excel_path = PATHS["grid_params_excel"]
        print(f"[LVGS-Engine] Excel path: {excel_path}")
        print(f"[LVGS-Engine] Excel exists: {os.path.exists(excel_path)}")

        if not os.path.exists(excel_path):
            lines = [l for l in f.getvalue().split('\n') if l.strip()]
            raise FileNotFoundError(f"Excel 参数文件不存在: {excel_path}")

        # --- 硬核测试模块引入与真正的环境变量 ---
        print(f"[LVGS-Engine] Worker Python Exe: {sys.executable}")
        try:
            import openpyxl
            print(f"[LVGS-Engine] Openpyxl loaded from: {openpyxl.__file__}")
        except Exception as oe:
            raise RuntimeError(f"底层完全无法导入 openpyxl! 真实报错={oe}\n{traceback.format_exc()}")

        try:
            import pandas as pd
            df_test = pd.read_excel(excel_path, sheet_name="EVStation", engine="openpyxl")
            print(f"[LVGS-Engine] EVStation rows: {len(df_test)}")
        except Exception as ev_err:
            tb = traceback.format_exc()
            with open("error_dump.txt", "w", encoding="utf-8") as err_f:
                err_f.write(f"EVStation 工作表读取失败: {ev_err}\n{tb}")
            raise RuntimeError(f"EVStation 工作表读取失败: {ev_err}\n{tb}")

        from power_grid_env import PowerGridEnv
        from evaluate_agents import evaluate_baseline, evaluate_rl_agent, discover_and_load_algorithms, \
            plot_and_save_results, plot_voltage_snapshots, plot_line_flow_snapshots_comparison
        from rl_normalization import load_eval_vecnormalize
    from config import PATHS

    # 系统默认使用两阶段潮流模式 (DistFlow + OpenDSS)
    use_two_stage = True
    print(f"[LVGS-Engine] Environment building for grid: {state.grid_model}, mode=two_stage")
    env = PowerGridEnv(gui_params=config.CORE_PARAMS, use_two_stage_flow=use_two_stage)
    seed = 0
    env.reset(seed=seed)

    # =========================================================
    # 全局缩放控制 (Global Multiplier Control)
    # 这一步极其重要，必须在节点级突变前进行全局参数的覆盖
    # =========================================================
    profile_mode = _apply_time_or_global_profiles(env, state)
    print(f"[LVGS-Engine] Applying scenario profile mode: {profile_mode}")

    # =========================================================
    # 场景扰动 (NodeOverrides)
    # 节点级负荷/PV/ESS/EV车位扩增由 PowerGridEnv.reset() 统一重建，
    # 避免 Baseline、RL reset、多模型对比走到不同场景。
    # =========================================================
    print(f"[LVGS-Engine] Node overrides bound to env reset path: {len(state.node_overrides)} item(s).")

    # =========================================================
    # 算法隔离调度 (Algorithmic Dispatching)
    # =========================================================
    metrics = {}
    time_series_log_for_this_seed = {}

    try:
        # 训练仍保持单算法模式：用户说“训练 SAC 1000 步”，就只训练 state.algo_name。
        if getattr(state, "execution_mode", "evaluate") == "train":
            algo = state.algo_name.strip().upper()
            if algo == "BASELINE" or algo == "":
                raise ValueError("Baseline 是优化/基线求解器，不需要训练。请指定 PPO/SAC/TD3/DDPG 等 RL 算法。")

            print(f"[LVGS-Engine] 用户明确指定进入训练模式，即将启动 {algo} 训练！")
            train_script = os.path.join(_CORE_PATH, f"train_{algo.lower()}.py")
            if not os.path.exists(train_script):
                raise FileNotFoundError(f"找不到对应的训练脚本：{train_script}")

            train_steps = state.rl_hyperparams.get("total_timesteps", 1000)
            print(f"[LVGS-Engine] 正在后台训练... (设定步数: {train_steps})")
            import subprocess
            import time
            env_copy = _prepare_training_subprocess_env(train_steps)
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            train_started_at = time.time()

            # 使用 Popen 进行流式读取，将进度实时打入前端
            process = subprocess.Popen(
                [sys.executable, train_script],
                env=env_copy,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=_CORE_PATH,
                creationflags=creationflags
            )

            captured_out = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    clean_line = _clean_process_log_line(line)
                    captured_out.append(clean_line)
                    if session_id and task_states and clean_line:
                        task_states[session_id]["logs"].append(clean_line)
                        if len(task_states[session_id]["logs"]) > 100:
                            task_states[session_id]["logs"] = task_states[session_id]["logs"][-100:]

            returncode = process.wait()
            if returncode != 0:
                saved_model = _find_recent_trained_model_after(
                    config.PATHS.get("models_dir", ""), algo, train_steps, train_started_at
                )
                if saved_model:
                    print(
                        f"[LVGS-Engine] 训练模型已保存，但子进程退出码为 {returncode}；"
                        f"按成功处理。模型: {saved_model}"
                    )
                    returncode = 0
            if returncode != 0:
                raise RuntimeError(f"训练失败！\nSTDOUT:\n{chr(10).join(captured_out[-50:])}")

            print(f"[LVGS-Engine] {algo} 训练完成！由于处于纯训练模式，本次任务结束。")
            metrics = {"is_train_only": True}
            return metrics, [l for l in f.getvalue().split('\n') if l.strip()] + captured_out[-20:]

        # 评估支持单模型，也支持 evaluation_targets 批量对比。
        targets = _normalize_evaluation_targets(state)
        is_comparison_request = bool(getattr(state, "evaluation_targets", None)) or len(targets) > 1
        # Baseline 可能会改写网格对象；尽量放到最后评估，避免影响 RL policy 的同场景评估。
        targets = [t for t in targets if t["algo"] != "BASELINE"] + [t for t in targets if t["algo"] == "BASELINE"]

        print(f"[LVGS-Engine] Evaluation targets: {targets}")
        needs_rl_models = any(str(t.get("algo", "")).strip().upper() not in {"", "BASELINE"} for t in targets)
        model_classes = discover_and_load_algorithms() if needs_rl_models else {}
        models_dir = PATHS.get("models_dir", "")
        comparison_metrics = {}
        current_scenario_fingerprint = build_scenario_fingerprint(_state_to_scenario_settings(state))

        for target in targets:
            target_algo = str(target.get("algo", "")).strip().upper()
            target_steps = _extract_steps(target.get("steps"))
            target_specific_model = str(target.get("specific_model_name", "") or "").strip()
            label = str(target.get("label") or target_algo).strip()

            if target_algo == "BASELINE" or target_algo == "":
                print(f"[LVGS-Engine] Running Baseline for comparison label={label}...")
                env.reset(seed=seed)
                target_metrics, ts_data = evaluate_baseline(config.CORE_PARAMS, seed, env.stations_list, env.grid,
                                                            use_two_stage=use_two_stage)
                comparison_metrics[label] = target_metrics
                time_series_log_for_this_seed[label] = ts_data
                continue

            print(
                f"[LVGS-Engine] Preparing RL inference environment for {label} ({target_algo}, steps={target_steps or 'latest'})...")
            model_path = _find_requested_model(
                models_dir,
                target_algo,
                target_steps,
                target_specific_model,
                scenario_fingerprint=current_scenario_fingerprint,
                require_scenario_match=True,
            )

            if model_path:
                print(
                    f"[LVGS-Engine] Target={label}; requested steps={target_steps if target_steps is not None else 'latest'}")
                print(f"[LVGS-Engine] Target={label}; selected model path={model_path}")

            if not model_path:
                requested = f"{target_steps} 步" if target_steps is not None else "当前场景"
                raise FileNotFoundError(
                    f"当前场景尚未训练可用于评估的 {target_algo} 模型（{requested}）。"
                    f"系统已阻止直接套用旧场景模型，也不会在评估模式下自动训练。"
                    f"请先把执行模式切换为 train 并训练 {target_algo}，训练完成后再评估。"
                )

            model_class = next((v for k, v in model_classes.items() if k.upper() == target_algo), None)
            if not model_class:
                raise ValueError(f"不受支持或未激活的 RL 算法: {target_algo}")

            manifest = _assert_model_matches_current_scenario(model_path, state, target_algo, label)

            print(f"[LVGS-Engine] Executing Stage 1 & 2 via RL Policy Network for {label}...")
            eval_normalizer = load_eval_vecnormalize(
                env,
                os.path.dirname(model_path),
                training_config=config.TRAINING_CONFIG,
                manifest=manifest,
            )
            model_env = eval_normalizer if eval_normalizer is not None else env
            model = model_class.load(model_path, env=model_env)
            target_metrics, ts_data = evaluate_rl_agent(
                model,
                env,
                seed,
                obs_normalizer=eval_normalizer,
            )
            target_metrics["_model_path"] = model_path
            target_metrics["_algo"] = target_algo
            target_metrics["_steps"] = target_steps if target_steps is not None else "latest"
            target_metrics["_scenario_fingerprint"] = manifest.get("scenario_fingerprint")
            comparison_metrics[label] = target_metrics
            time_series_log_for_this_seed[label] = ts_data

        if not comparison_metrics:
            raise RuntimeError("没有任何模型被成功评估。")

        if is_comparison_request:
            metrics = {"is_comparison": True, "comparison_metrics": comparison_metrics}
        else:
            metrics = next(iter(comparison_metrics.values()))
    except Exception as exec_err:
        raise RuntimeError(f"{str(exec_err)}\n\n[SOLVER & STDOUT LOGS]:\n{f.getvalue()}") from exec_err

    print(f"[LVGS-Engine] Generating result charts in backend...")
    if time_series_log_for_this_seed and any(v for v in time_series_log_for_this_seed.values() if v):
        try:
            plot_and_save_results(time_series_log_for_this_seed, seed, config.CORE_PARAMS)
            plot_voltage_snapshots(time_series_log_for_this_seed, seed, config.CORE_PARAMS)
            plot_line_flow_snapshots_comparison(time_series_log_for_this_seed, seed, config.CORE_PARAMS)
        except Exception as plot_err:
            print(f"[WARNING] Chart generation failed: {plot_err}")

    print(f"[LVGS-Engine] Aggregating results. Total mapped cost: {metrics.get('总成本', 0):.2f}")
    return metrics, [l for l in f.getvalue().split('\n') if l.strip()]
