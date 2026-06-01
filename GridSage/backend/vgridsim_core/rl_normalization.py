import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


VECNORMALIZE_FILENAME = "vecnormalize.pkl"


def _positive_float(value, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0 else float(default)


def get_vecnormalize_config(training_config: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
    cfg = training_config or {}
    vec_cfg = dict(cfg.get("vec_normalize", {}) or {})
    use_vec_normalize = bool(cfg.get("use_vec_normalize", True))
    return use_vec_normalize, {
        "norm_obs": bool(vec_cfg.get("norm_obs", True)),
        "norm_reward": bool(vec_cfg.get("norm_reward", True)),
        "clip_obs": _positive_float(vec_cfg.get("clip_obs", 10.0), 10.0),
        "clip_reward": _positive_float(vec_cfg.get("clip_reward", 10.0), 10.0),
    }


def make_dummy_vec_env(env_raw):
    monitored = Monitor(env_raw)
    return DummyVecEnv([lambda: monitored])


def wrap_training_env(env_raw, gamma: float, training_config: Optional[Dict[str, Any]] = None):
    use_vec_normalize, vec_cfg = get_vecnormalize_config(training_config)
    vec_env = make_dummy_vec_env(env_raw)
    if not use_vec_normalize:
        print("[RL-Normalization] VecNormalize disabled by config.")
        return vec_env

    print(
        "[RL-Normalization] VecNormalize enabled: "
        f"norm_obs={vec_cfg['norm_obs']}, norm_reward={vec_cfg['norm_reward']}"
    )
    return VecNormalize(
        vec_env,
        norm_obs=vec_cfg["norm_obs"],
        norm_reward=vec_cfg["norm_reward"],
        clip_obs=vec_cfg["clip_obs"],
        clip_reward=vec_cfg["clip_reward"],
        gamma=float(gamma),
    )


def wrap_callback_eval_env(env_raw, gamma: float, training_config: Optional[Dict[str, Any]] = None):
    use_vec_normalize, vec_cfg = get_vecnormalize_config(training_config)
    vec_env = make_dummy_vec_env(env_raw)
    if not use_vec_normalize:
        return vec_env

    eval_env = VecNormalize(
        vec_env,
        norm_obs=vec_cfg["norm_obs"],
        norm_reward=False,
        clip_obs=vec_cfg["clip_obs"],
        clip_reward=vec_cfg["clip_reward"],
        gamma=float(gamma),
        training=False,
    )
    eval_env.norm_reward = False
    return eval_env


def is_vecnormalize(env) -> bool:
    return isinstance(env, VecNormalize)


def find_vecnormalize(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, VecNormalize):
            return current
        current = getattr(current, "venv", None)
    return None


def save_vecnormalize_stats(env, model_dir: str) -> Optional[str]:
    vecnormalize = find_vecnormalize(env)
    if vecnormalize is None:
        return None

    path = os.path.join(model_dir, VECNORMALIZE_FILENAME)
    vecnormalize.save(path)
    print(f"[RL-Normalization] Saved VecNormalize stats to: {os.path.abspath(path)}")
    return VECNORMALIZE_FILENAME


def resolve_vecnormalize_path(model_dir: str, manifest: Optional[Dict[str, Any]] = None) -> str:
    manifest = manifest or {}
    rel_path = manifest.get("vecnormalize_path") or VECNORMALIZE_FILENAME
    if os.path.isabs(str(rel_path)):
        return str(rel_path)
    return os.path.join(model_dir, str(rel_path))


def load_eval_vecnormalize(
    env_raw,
    model_dir: str,
    training_config: Optional[Dict[str, Any]] = None,
    manifest: Optional[Dict[str, Any]] = None,
):
    use_vec_normalize, _ = get_vecnormalize_config(training_config)
    manifest = manifest or {}
    manifest_requires_vecnormalize = bool(manifest.get("use_vec_normalize", False))
    vecnormalize_path = resolve_vecnormalize_path(model_dir, manifest)

    if not use_vec_normalize and not manifest_requires_vecnormalize:
        return None

    if os.path.exists(vecnormalize_path):
        eval_env = make_dummy_vec_env(env_raw)
        eval_env = VecNormalize.load(vecnormalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        print(f"[RL-Normalization] Loaded VecNormalize stats from: {os.path.abspath(vecnormalize_path)}")
        return eval_env

    if manifest_requires_vecnormalize:
        raise FileNotFoundError(
            f"model manifest requires VecNormalize stats, but file was not found: {vecnormalize_path}"
        )

    print(
        "[RL-Normalization] VecNormalize stats not found for this model; "
        "continuing for legacy compatibility, but evaluation may be inconsistent with the new training flow."
    )
    return None


def normalize_observation(obs, obs_normalizer=None):
    if obs_normalizer is None:
        return obs
    obs_array = np.asarray(obs, dtype=np.float32)
    normalized = obs_normalizer.normalize_obs(obs_array.copy())
    return np.asarray(normalized, dtype=np.float32)


def predict_with_optional_normalization(model, obs, obs_normalizer=None, deterministic: bool = True):
    model_obs = normalize_observation(obs, obs_normalizer)
    return model.predict(model_obs, deterministic=deterministic)


def unwrap_power_grid_env(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "params") and hasattr(current, "total_spots"):
            return current
        if hasattr(current, "envs") and current.envs:
            current = current.envs[0]
            continue
        if hasattr(current, "venv"):
            current = current.venv
            continue
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    return env

