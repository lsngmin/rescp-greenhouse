"""Shared helpers for the GreenLight-Gym runs (conda env glgym): environment construction and
one-step recording of the same nine variables as the measured datasets plus the controller set points."""
import os
from pathlib import Path
import numpy as np
import yaml
from gl_gym.environments.greenlight_env import GreenLightEnv
from gl_gym.components.rule_based import RuleBasedController
from gl_gym.core.types import StepContext
from gl_gym.environments.utils import satVp, load_weather_data
from gl_gym.components.weather import WeatherRepository

ROOT = Path(__file__).resolve().parents[1]
GLGYM = ROOT / "third_party" / "GreenLight-Gym"          # weather files and configs/agents/rule_based.yml
CFG = ROOT / "simulation" / "configs" / "envs"            # GreenLightEnv.yml used in the paper


# same as the three functions in GreenLight-Gym RL/utils.py, without the wandb / stable-baselines3 imports
def load_env_params(env_id, path):
    with open(os.path.join(path, env_id + ".yml")) as f:
        return yaml.load(f, Loader=yaml.FullLoader)[env_id]


def build_env_kwargs(env_kwargs):
    env_kwargs = env_kwargs.copy(); w = env_kwargs.pop("weather_repository_kwargs")
    env_kwargs["weather_repository"] = WeatherRepository(weather_data_dir=w["weather_data_dir"], load_weather_data_fn=load_weather_data)
    return env_kwargs, env_kwargs.pop("eval_scenarios")


def load_model_hyperparams(algorithm, env_id):
    with open(os.path.join(GLGYM, "configs", "agents", algorithm + ".yml")) as f:
        return yaml.load(f, Loader=yaml.FullLoader)[env_id]


def make(start_day=226, year=2011, season=115):
    os.chdir(GLGYM)  # the weather directory in GreenLightEnv.yml is relative to the GreenLight-Gym root
    kw = load_env_params("GreenLightEnv", CFG); kw, sc = build_env_kwargs(kw)
    kw["normalize_actions"] = False; kw["season_length"] = season
    rb = load_model_hyperparams("rule_based", "GreenLightEnv")
    env = GreenLightEnv(**kw)
    scen = {"location": "Amsterdam", "growth_year": year, "start_day": start_day}
    return env, rb, scen


def ctx_of(env):
    return StepContext(t=env.timestep, dt=env.dt, Np=env.Np, x_prev=env.x_prev, x=env.x, u=env.u, p=env.p,
                       d=env.weather_data, hour_of_day=env.hour_of_day, day_of_year=env.day_of_year)


def record(env, ctl):
    """Current state in the nine-variable format. Set points: temperature ventilation threshold (heat set point +
    dead zone) and humidity ventilation threshold (rh_max)."""
    x, d = env.x, env.weather_data[env.timestep]
    t_in = float(x[2]); rh_in = float(100 * x[15] / satVp(x[2]))
    t_out = float(d[1]); rh_out = float(100 * d[2] / satVp(d[1]))
    h = env.hour_of_day % 24
    # day/night decision as in the rule-based controller -> heating set point
    lamp = (ctl.lamps_on < h < ctl.lamps_off) if ctl.lamps_on <= ctl.lamps_off else (h > ctl.lamps_on or h < ctl.lamps_off)
    is_day = max(float(lamp and d[7] < ctl.lamp_rad_sum_limit), float(d[8]))
    heat_sp = is_day * ctl.temp_setpoint_day + (1 - is_day) * ctl.temp_setpoint_night
    return {"T_in": t_in, "RH_in": rh_in, "T_out": t_out, "RH_out": rh_out, "WS_out": float(d[4]),
            "Light_out": float(d[0]), "Vent": float(env.u[3]) * 100,
            "hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
            "T_vent_sp": heat_sp + ctl.heat_deadzone, "RH_vent_sp": ctl.rh_max}
