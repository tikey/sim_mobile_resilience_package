#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimMobileResilience — експериментальний стенд:
- три рівні D/P/R, часткова спостережуваність, переривчаста зв’язність
- локальні інваріанти та порушення
- тропічне (max–times) поширення індукованих порушень
- контрактні "залишки" та віконна узгодженість
- інтегральний показник B* (структура + контракти + ризик (CVaR))
- дві політики: reactive, myopic

CLI:
    python sim_mobile_resilience.py --policy both --T 420 --seed 7 --outdir out

Авторське попередження: це референсний код-стенд для досліджень; не оптимізований під продуктивність.
"""
import argparse
import math
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# YAML параметри (необов’язково)
try:
    import yaml
except Exception:
    yaml = None


# ---------------- Параметри моделі ----------------
@dataclass
class ModelParams:
    # Нормування рівнів
    s_D: float = 1.0
    s_P: float = 1.0
    s_R: float = 1.0
    # Пороги "нормальних" режимів (локальні інваріанти)
    tau_AoI: float = 1.0     # поріг валідності даних (чим менше, тим краще)
    theta_resp: float = 1.0  # поріг часу відгуку процесів
    slack_R: float = 0.0     # потрібний запас ресурсів (>=0)
    # Радіус локалізації
    r_star: int = 2
    # Згладжування локальних оцінок
    lambda_D: float = 0.2
    lambda_P: float = 0.2
    lambda_R: float = 0.2
    # Тропічні коефіцієнти затухання шляхів
    mu1: float = 1.0
    mu2: float = 0.6
    mu3: float = 0.4
    # Параметри ризикового блоку (CVaR)
    cvar_alpha: float = 0.95
    cvar_window: int = 40
    s_SLO: float = 1.0
    # Пороги інтерпретації B*
    theta_ok: float = 0.25
    theta_warn: float = 0.5
    # Ваги блоків B*
    w_str: float = 1.0
    w_ctr: float = 1.0
    w_risk: float = 1.0
    # Маскування доступності (частоти появи вікон)
    p_alpha_DP: float = 0.75
    p_alpha_PR: float = 0.65
    p_alpha_RD: float = 0.55
    # Саторування підйомів
    sat_scale_D: float = 1.0
    sat_scale_P: float = 1.0
    sat_scale_R: float = 1.0
    # Підсилення (базові)
    base_g_DP: float = 0.8
    base_g_PR: float = 0.9
    base_g_RD: float = 0.7
    # Віконна компенсація (довжина)
    window_l: int = 6
    # Згладження B*
    rho_B: float = 0.05


@dataclass
class EnvState:
    alpha_DP: int
    alpha_PR: int
    alpha_RD: int
    bw: float
    cpu: float
    energy: float


@dataclass
class SystemState:
    AoI: float
    resp: float
    res_def: float
    vD_bar: float = 0.0
    vP_bar: float = 0.0
    vR_bar: float = 0.0
    tD: float = 0.0
    tP: float = 0.0
    tR: float = 0.0
    R_DP: float = 0.0
    R_PR: float = 0.0
    R_RD: float = 0.0
    Bstar: float = 0.0
    Bstar_smooth: float = 0.0
    SLO_def: float = 0.0
    cost_u: float = 0.0
    action: str = "idle"


# ---------------- Допоміжні обчислення ----------------
def local_violations(state: SystemState, mp: ModelParams) -> Tuple[float, float, float]:
    vD = max(0.0, state.AoI - mp.tau_AoI)
    vP = max(0.0, state.resp - mp.theta_resp)
    vR = max(0.0, state.res_def - mp.slack_R)
    return vD, vP, vR

def exp_smooth(prev: float, val: float, lam: float) -> float:
    return (1 - lam) * val + lam * prev

def sat(z: float, s: float) -> float:
    if s <= 0: return z
    return s * (1.0 - math.exp(-z / s))

def build_G(env: EnvState, mp: ModelParams) -> np.ndarray:
    gDP = env.alpha_DP * mp.base_g_DP * (0.5 + 0.5 * env.bw)
    gPR = env.alpha_PR * mp.base_g_PR * (0.5 + 0.5 * min(env.cpu, env.bw))
    gRD = env.alpha_RD * mp.base_g_RD * (0.5 + 0.5 * min(env.energy, env.cpu))
    G = np.array([
        [0.0,       gPR*0.25, 0.0],
        [gDP,       0.0,      0.4*gRD],
        [0.0,       0.5*gPR,  0.0],
    ], dtype=float)
    return G

def tropical_path_contrib(G: np.ndarray, vbar: np.ndarray, mp: ModelParams) -> Tuple[np.ndarray, np.ndarray]:
    z1 = np.max(G * vbar.reshape(1, -1), axis=1)
    def tropical_matmul(A, B):
        n = A.shape[0]; m = B.shape[1]
        out = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                out[i, j] = np.max(A[i, :] * B[:, j])
        return out
    G2 = tropical_matmul(G, G)
    z2 = np.max(G2 * vbar.reshape(1, -1), axis=1)
    z_path = mp.mu1 * z1 + mp.mu2 * z2
    t = np.maximum(z1, z2)
    return t, z_path

def contract_residuals(state: SystemState, env: EnvState, mp: ModelParams) -> Tuple[float, float, float]:
    A_DP = max(0.0, state.AoI - mp.tau_AoI)
    G_D  = env.alpha_DP * env.bw
    R_DP = max(0.0, A_DP - G_D) * env.alpha_DP
    A_PR = max(0.0, state.resp - mp.theta_resp)
    G_R  = env.alpha_PR * min(env.cpu, env.bw)
    R_PR = max(0.0, A_PR - G_R) * env.alpha_PR
    A_RD = max(0.0, state.res_def - mp.slack_R)
    G_R2 = env.alpha_RD * env.energy
    R_RD = max(0.0, A_RD - G_R2) * env.alpha_RD
    return R_DP, R_PR, R_RD

def cvar(values: List[float], alpha: float) -> float:
    if not values:
        return 0.0
    xs = np.array(values, dtype=float)
    q = np.quantile(xs, alpha)
    tail = xs[xs >= q]
    if len(tail) == 0:
        return 0.0
    return float(np.mean(tail))

def compute_Bstar(state: SystemState, env: EnvState, mp: ModelParams, hist_SLO: List[float]) -> Tuple[float, Dict[str, float]]:
    z = np.array([
        state.vD_bar / mp.s_D, state.vP_bar / mp.s_P, state.vR_bar / mp.s_R,
        state.tD / mp.s_D,     state.tP / mp.s_P,     state.tR / mp.s_R
    ])
    B_str = float(np.max(z))
    R_vec = np.array([state.R_DP, state.R_PR, state.R_RD])
    cpl = np.array([
        min(state.vD_bar / mp.s_D, state.tP / mp.s_P),
        min(state.vP_bar / mp.s_P, state.tR / mp.s_R),
        min(state.vR_bar / mp.s_R, state.tD / mp.s_D),
    ])
    B_ctr = float(max(np.max(R_vec), np.max(cpl)))
    window = hist_SLO[-mp.cvar_window:] if len(hist_SLO) >= mp.cvar_window else hist_SLO
    B_risk = cvar(window, mp.cvar_alpha) / max(1e-9, mp.s_SLO)
    Bstar = max(mp.w_str * B_str, mp.w_ctr * B_ctr, mp.w_risk * B_risk)
    return Bstar, {"B_str": B_str, "B_ctr": B_ctr, "B_risk": B_risk}


# ---------------- Політики ----------------
def policy_reactive(state: SystemState, env: EnvState, mp: ModelParams) -> Tuple[str, float]:
    scores = {
        "replicate": state.vD_bar + state.tP + state.R_DP,
        "degrade":   state.vP_bar + state.tR + state.R_PR,
        "realloc":   state.vR_bar + state.tD + state.R_RD,
        "idle":      0.01
    }
    action = max(scores.items(), key=lambda kv: kv[1])[0]
    cost = {"replicate": 0.7, "degrade": 0.3, "realloc": 0.5, "idle": 0.0}[action]
    return action, cost

def policy_myopic(state: SystemState, env: EnvState, mp: ModelParams) -> Tuple[str, float]:
    candidates = ["replicate", "degrade", "realloc", "idle"]
    def delta_after(action: str) -> Tuple[float, float, float]:
        dD = dP = dR = 0.0
        if action == "replicate" and env.alpha_DP:
            dD = -0.5 * env.bw
            dP = -0.2 * env.bw
        if action == "degrade":
            dP = -0.4
        if action == "realloc" and env.alpha_PR:
            dR = -0.5 * min(env.cpu, env.energy)
            dP = -0.2 * min(env.cpu, env.bw)
        return dD, dP, dR
    best = ("idle", math.inf, 0.0)
    for a in candidates:
        dD, dP, dR = delta_after(a)
        AoI = max(0.0, state.AoI + dD)
        resp = max(0.0, state.resp + dP)
        resd = max(0.0, state.res_def + dR)
        vD = max(0.0, AoI - mp.tau_AoI)
        vP = max(0.0, resp - mp.theta_resp)
        vR = max(0.0, resd - mp.slack_R)
        vbar = np.array([
            exp_smooth(state.vD_bar, vD, mp.lambda_D) / mp.s_D,
            exp_smooth(state.vP_bar, vP, mp.lambda_P) / mp.s_P,
            exp_smooth(state.vR_bar, vR, mp.lambda_R) / mp.s_R
        ])
        G = build_G(env, mp)
        t, zpath = tropical_path_contrib(G, vbar, mp)
        R_dp, R_pr, R_rd = state.R_DP, state.R_PR, state.R_RD
        B_str = float(np.max(np.concatenate([vbar, t / np.array([mp.s_D, mp.s_P, mp.s_R])])))
        B_ctr = float(max(R_dp, R_pr, R_rd))
        B_pred = max(mp.w_str * B_str, mp.w_ctr * B_ctr, mp.w_risk * state.SLO_def)
        cost = {"replicate": 0.7, "degrade": 0.3, "realloc": 0.5, "idle": 0.0}[a]
        if B_pred + 0.05 * cost < best[1]:
            best = (a, B_pred + 0.05 * cost, cost)
    return best[0], best[2]


# ---------------- Динаміка середовища та системи ----------------
def sample_env(mp: ModelParams) -> EnvState:
    alpha_DP = 1 if np.random.rand() < mp.p_alpha_DP else 0
    alpha_PR = 1 if np.random.rand() < mp.p_alpha_PR else 0
    alpha_RD = 1 if np.random.rand() < mp.p_alpha_RD else 0
    bw = np.clip(np.random.beta(2, 3) + 0.4 * alpha_DP, 0, 1)
    cpu = np.clip(np.random.beta(2, 2) + 0.3 * alpha_PR, 0, 1)
    energy = np.clip(np.random.beta(2, 4) + 0.3 * alpha_RD, 0, 1)
    return EnvState(alpha_DP, alpha_PR, alpha_RD, float(bw), float(cpu), float(energy))

def apply_action(state: SystemState, env: EnvState, mp: ModelParams, action: str):
    if action == "replicate" and env.alpha_DP:
        state.AoI = max(0.0, state.AoI - 0.6 * env.bw)
        state.resp = max(0.0, state.resp - 0.15 * env.bw)
        state.res_def = max(0.0, state.res_def - 0.1 * min(env.energy, env.cpu))
    elif action == "degrade":
        state.resp = max(0.0, state.resp - 0.4)
    elif action == "realloc" and env.alpha_PR:
        state.res_def = max(0.0, state.res_def - 0.6 * min(env.cpu, env.energy))
        state.resp = max(0.0, state.resp - 0.2 * min(env.cpu, env.bw))

def natural_drift(state: SystemState):
    state.AoI += np.random.gamma(1.0, 0.08)
    state.resp = max(0.0, state.resp + np.random.normal(0.02, 0.05))
    state.res_def = max(0.0, state.res_def + np.random.normal(0.0, 0.04))


# ---------------- Симуляція ----------------
def run_sim(policy_name: str, T: int = 400, seed: int = 123) -> Tuple[pd.DataFrame, Dict]:
    random.seed(seed); np.random.seed(seed)
    mp = ModelParams()
    st = SystemState(AoI=0.9, resp=0.9, res_def=0.1)
    hist_SLO: List[float] = []
    rows = []
    for t in range(T):
        env = sample_env(mp)
        vD, vP, vR = local_violations(st, mp)
        st.vD_bar = exp_smooth(st.vD_bar, vD, mp.lambda_D)
        st.vP_bar = exp_smooth(st.vP_bar, vP, mp.lambda_P)
        st.vR_bar = exp_smooth(st.vR_bar, vR, mp.lambda_R)
        vbar_vec = np.array([st.vD_bar / mp.s_D, st.vP_bar / mp.s_P, st.vR_bar / mp.s_R])
        G = build_G(env, mp)
        t_induced, z_path = tropical_path_contrib(G, vbar_vec, mp)
        st.tD, st.tP, st.tR = float(t_induced[0]), float(t_induced[1]), float(t_induced[2])
        st.R_DP, st.R_PR, st.R_RD = contract_residuals(st, env, mp)
        st.SLO_def = max(0.0, st.resp - mp.theta_resp)
        Bstar, parts = compute_Bstar(st, env, mp, hist_SLO)
        st.Bstar = Bstar
        st.Bstar_smooth = (1 - mp.rho_B) * Bstar + mp.rho_B * st.Bstar_smooth
        if policy_name == "reactive":
            action, cost = policy_reactive(st, env, mp)
        elif policy_name == "myopic":
            action, cost = policy_myopic(st, env, mp)
        else:
            action, cost = ("idle", 0.0)
        st.action = action; st.cost_u = cost
        apply_action(st, env, mp, action)
        natural_drift(st)
        hist_SLO.append(st.SLO_def)
        rows.append({
            "t": t, "alpha_DP": env.alpha_DP, "alpha_PR": env.alpha_PR, "alpha_RD": env.alpha_RD,
            "bw": env.bw, "cpu": env.cpu, "energy": env.energy,
            "AoI": st.AoI, "resp": st.resp, "res_def": st.res_def,
            "vD_bar": st.vD_bar, "vP_bar": st.vP_bar, "vR_bar": st.vR_bar,
            "tD": st.tD, "tP": st.tP, "tR": st.tR,
            "R_DP": st.R_DP, "R_PR": st.R_PR, "R_RD": st.R_RD,
            "Bstar": st.Bstar, "Bstar_smooth": st.Bstar_smooth,
            "SLO_def": st.SLO_def, "action": st.action, "cost_u": st.cost_u,
            "B_str": parts["B_str"], "B_ctr": parts["B_ctr"], "B_risk": parts["B_risk"],
        })
    df = pd.DataFrame(rows)
    mp2 = mp
    M_B1 = df["Bstar"].mean()
    M_B2_warn = ((df["Bstar"] > mp2.theta_ok) & (df["Bstar"] <= mp2.theta_warn)).mean()
    M_B2_crit = (df["Bstar"] > mp2.theta_warn).mean()
    M_B3 = (df["Bstar"] - mp2.theta_ok).clip(lower=0).sum()
    M_SLO1 = 1.0 - (df["SLO_def"] > 0).mean()
    M_SLO2 = df["SLO_def"].sum()
    M_SLO3 = cvar(df["SLO_def"].tolist(), mp2.cvar_alpha)
    M_CTR1 = (df["R_DP"] + df["R_PR"] + df["R_RD"]).sum()
    M_CTR2 = max(df["R_DP"].max(), df["R_PR"].max(), df["R_RD"].max())
    M_LOC1 = float(((df["tD"] > df["vD_bar"]) | (df["tP"] > df["vP_bar"]) | (df["tR"] > df["vR_bar"])).mean())
    M_u1 = df["cost_u"].sum()
    M_u2 = (df["action"] != "idle").mean()
    M_u3 = (df["action"].shift(-1) != df["action"]).sum() / len(df)
    summary = {
        "policy": policy_name, "T": T,
        "M_B1_avg": M_B1, "M_B2_warn_frac": M_B2_warn, "M_B2_crit_frac": M_B2_crit, "M_B3_area": M_B3,
        "M_SLO1_rate": M_SLO1, "M_SLO2_sum": M_SLO2, "M_SLO3_cvar": M_SLO3,
        "M_CTR1_sum": M_CTR1, "M_CTR2_max": M_CTR2,
        "M_LOC1_cascade_frac": M_LOC1,
        "M_u1_total_cost": M_u1, "M_u2_intervention_rate": M_u2, "M_u3_switching": M_u3,
    }
    return df, summary


# ----------------- Візуалізація -----------------
def plot_timeseries_Bstar(df_react: pd.DataFrame, df_myop: pd.DataFrame, mp: ModelParams, outdir: str):
    plt.figure()
    plt.plot(df_react["t"], df_react["Bstar"], label="B* (reactive)")
    plt.plot(df_myop["t"], df_myop["Bstar"], label="B* (myopic)")
    plt.axhline(mp.theta_ok, linestyle="--", label="θ_ok")
    plt.axhline(mp.theta_warn, linestyle="--", label="θ_warn")
    plt.xlabel("t"); plt.ylabel("B*"); plt.title("Інтегральний показник B* у часі (дві політики)")
    plt.legend()
    p = f"{outdir}/fig_Bstar_time.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()

def plot_blocks_bar(df_react: pd.DataFrame, df_myop: pd.DataFrame, outdir: str):
    plt.figure()
    means_blocks = pd.DataFrame({
        "B_str": [df_react["B_str"].mean(), df_myop["B_str"].mean()],
        "B_ctr": [df_react["B_ctr"].mean(), df_myop["B_ctr"].mean()],
        "B_risk": [df_react["B_risk"].mean(), df_myop["B_risk"].mean()],
    }, index=["reactive", "myopic"])
    means_blocks.plot(kind="bar")
    plt.title("Середні значення субпоказників (структура/контракти/ризик)")
    plt.ylabel("середнє значення")
    p = f"{outdir}/fig_blocks_bar.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()

def plot_SLO_time(df: pd.DataFrame, outdir: str, tag: str):
    plt.figure()
    plt.plot(df["t"], df["SLO_def"])
    plt.xlabel("t"); plt.ylabel("дефіцит SLO"); plt.title(f"Динаміка дефіциту SLO ({tag})")
    p = f"{outdir}/fig_slo_time_{tag}.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()

def plot_contracts_time(df: pd.DataFrame, outdir: str, tag: str):
    plt.figure()
    plt.plot(df["t"], df[["R_DP","R_PR","R_RD"]].max(axis=1))
    plt.xlabel("t"); plt.ylabel("максимальний залишок контракту"); plt.title(f"Максимальний контрактний борг ({tag})")
    p = f"{outdir}/fig_contracts_time_{tag}.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()

def plot_hist_Bstar(df_react: pd.DataFrame, df_myop: pd.DataFrame, outdir: str):
    plt.figure()
    plt.hist(df_react["Bstar"], bins=30, alpha=0.6, label="reactive")
    plt.hist(df_myop["Bstar"], bins=30, alpha=0.6, label="myopic")
    plt.xlabel("B*"); plt.ylabel("частота"); plt.title("Розподіл значень B* (дві політики)")
    plt.legend()
    p = f"{outdir}/fig_Bstar_hist.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()


# ----------------- Утиліти -----------------
def load_params_from_yaml(path: str, mp: ModelParams) -> ModelParams:
    if (yaml is None) or (not path):
        return mp
    try:
        with open(path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f) or {}
        for k, v in conf.items():
            if hasattr(mp, k):
                setattr(mp, k, v)
    except Exception:
        pass
    return mp

def main():
    ap = argparse.ArgumentParser(description="SimMobileResilience — стенд міжрівневої живучості (розд. 4.6)")
    ap.add_argument("--policy", default="both", choices=["reactive","myopic","both"], help="яку політику запускати")
    ap.add_argument("--T", type=int, default=420, help="довжина симуляції")
    ap.add_argument("--seed", type=int, default=7, help="випадкове зерно")
    ap.add_argument("--outdir", default="out", help="директорія для артефактів")
    ap.add_argument("--params", default="params.yaml", help="YAML з параметрами моделі (необов’язково)")
    ap.add_argument("--no-plots", action="store_true", help="не будувати графіків")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    mp = ModelParams()
    mp = load_params_from_yaml(args.params, mp)

    import os
    os.makedirs(args.outdir, exist_ok=True)

    if args.policy in ("reactive","both"):
        df_react, sum_react = run_sim("reactive", T=args.T, seed=args.seed)
        df_react.to_csv(f"{args.outdir}/sim_results_reactive.csv", index=False)
        with open(f"{args.outdir}/sim_summary_reactive.json","w",encoding="utf-8") as f:
            json.dump(sum_react, f, ensure_ascii=False, indent=2)

    if args.policy in ("myopic","both"):
        df_myop, sum_myop = run_sim("myopic", T=args.T, seed=args.seed)
        df_myop.to_csv(f"{args.outdir}/sim_results_myopic.csv", index=False)
        with open(f"{args.outdir}/sim_summary_myopic.json","w",encoding="utf-8") as f:
            json.dump(sum_myop, f, ensure_ascii=False, indent=2)

    # Плоти
    if not args.no_plots:
        if args.policy == "both":
            plot_timeseries_Bstar(df_react, df_myop, ModelParams(), args.outdir)
            plot_blocks_bar(df_react, df_myop, args.outdir)
            plot_hist_Bstar(df_react, df_myop, args.outdir)
            plot_SLO_time(df_myop, args.outdir, "myopic")
            plot_contracts_time(df_react, args.outdir, "reactive")
        elif args.policy == "reactive":
            plot_timeseries_Bstar(df_react, df_react, ModelParams(), args.outdir)
            plot_blocks_bar(df_react, df_react, args.outdir)
            plot_hist_Bstar(df_react, df_react, args.outdir)
            plot_SLO_time(df_react, args.outdir, "reactive")
            plot_contracts_time(df_react, args.outdir, "reactive")
        else:
            plot_timeseries_Bstar(df_myop, df_myop, ModelParams(), args.outdir)
            plot_blocks_bar(df_myop, df_myop, args.outdir)
            plot_hist_Bstar(df_myop, df_myop, args.outdir)
            plot_SLO_time(df_myop, args.outdir, "myopic")
            plot_contracts_time(df_myop, args.outdir, "myopic")

    # Зведений summary
    summary = {}
    if args.policy in ("reactive","both"):
        summary["reactive"] = json.load(open(f"{args.outdir}/sim_summary_reactive.json","r",encoding="utf-8"))
    if args.policy in ("myopic","both"):
        summary["myopic"] = json.load(open(f"{args.outdir}/sim_summary_myopic.json","r",encoding="utf-8"))
    with open(f"{args.outdir}/sim_summary_all.json","w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Готово. Артефакти у:", args.outdir)

if __name__ == "__main__":
    main()
