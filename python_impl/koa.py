"""Kepler Optimization Algorithm (KOA) translated from MATLAB to Python."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from .objective_function import ObjectiveResult, objective_function


@dataclass
class KOAResult:
    best_score: float
    best_position: np.ndarray
    curve: np.ndarray
    best_predictions: np.ndarray
    best_model: torch.nn.Module
    best_metrics: dict


_EPS = 1e-12


def _initialization(
    agents: int,
    dim: int,
    upper_bounds: np.ndarray,
    lower_bounds: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.random((agents, dim)) * (upper_bounds - lower_bounds) + lower_bounds


def koa(
    search_agents: int,
    t_max: int,
    upper_bounds: Sequence[float],
    lower_bounds: Sequence[float],
    dim: int,
    *,
    data_path: str | None = None,
    device: torch.device | None = None,
    random_state: int | None = None,
) -> KOAResult:
    """Run the Kepler Optimization Algorithm using the objective function."""

    rng = np.random.default_rng(random_state)
    ub = np.asarray(upper_bounds, dtype=float)
    lb = np.asarray(lower_bounds, dtype=float)

    orbital = rng.random(search_agents)
    T = np.abs(rng.normal(size=search_agents))
    positions = _initialization(search_agents, dim, ub, lb, rng)

    sun_pos = np.zeros(dim)
    sun_score = math.inf

    pl_fit = np.zeros(search_agents)
    predictions: List[np.ndarray] = [np.zeros(24, dtype=np.float32) for _ in range(search_agents)]
    nets: List[torch.nn.Module] = [None for _ in range(search_agents)]  # type: ignore
    infos: List[dict] = [{} for _ in range(search_agents)]

    for i in range(search_agents):
        result = objective_function(positions[i], data_path=data_path, device=device)
        pl_fit[i] = result.loss_metric
        predictions[i] = result.predictions
        nets[i] = result.model
        infos[i] = {
            "history": result.history,
            "metrics": result.metrics,
            "position": positions[i].copy(),
        }
        if pl_fit[i] < sun_score:
            sun_score = pl_fit[i]
            sun_pos = positions[i].copy()
            best_prediction = result.predictions
            best_net = result.model
            best_info = infos[i]

    tc = 3
    m0 = 0.1
    lambda_val = 15
    t = 0

    while t < t_max:
        order = np.sort(pl_fit)
        worst_fitness = order[-1]
        denom = np.sum(pl_fit - worst_fitness) + _EPS
        M = m0 * math.exp(-lambda_val * (t / max(t_max, 1)))

        R = np.linalg.norm(sun_pos - positions, axis=1)
        Rnorm = (R - R.min()) / (R.max() - R.min() + _EPS)

        MS = np.zeros(search_agents)
        m = np.zeros(search_agents)
        MSnorm = np.zeros(search_agents)
        Mnorm = np.zeros(search_agents)

        for i in range(search_agents):
            MS[i] = rng.random() * (sun_score - worst_fitness) / denom
            m[i] = (pl_fit[i] - worst_fitness) / denom

        MSnorm = (MS - MS.min()) / (MS.max() - MS.min() + _EPS)
        Mnorm = (m - m.min()) / (m.max() - m.min() + _EPS)

        Fg = orbital * M * (MSnorm * Mnorm) / (Rnorm * Rnorm + _EPS) + rng.random(search_agents)
        a1 = np.array(
            [
                rng.random()
                * (T[i] ** 2 * (M * (MS[i] + m[i]) / (4 * math.pi * math.pi))) ** (1.0 / 3.0)
                for i in range(search_agents)
            ]
        )

        for i in range(search_agents):
            a2 = -1 - (math.remainder(t, t_max / tc) / (t_max / tc + _EPS))
            n = (a2 - 1) * rng.random() + 1
            a = rng.integers(0, search_agents)
            b = rng.integers(0, search_agents)
            rd = rng.random(dim)
            r = rng.random()
            u1 = rd < r
            original_pos = positions[i].copy()

            if rng.random() < rng.random():
                h = 1.0 / math.exp(n * rng.normal())
                xm = (positions[b] + sun_pos + positions[i]) / 3.0
                positions[i] = (
                    positions[i] * u1.astype(float)
                    + (xm + h * (xm - positions[a])) * (~u1).astype(float)
                )
            else:
                f = 1 if rng.random() < 0.5 else -1
                L = (
                    M
                    * (MS[i] + m[i])
                    * abs((2.0 / (R[i] + _EPS)) - (1.0 / (a1[i] + _EPS)))
                ) ** 0.5
                U = rd > rng.random(dim)

                if Rnorm[i] < 0.5:
                    M_temp = rng.random(dim) * (1 - r) + r
                    l = L * M_temp * U
                    Mv = rng.random(dim) * (1 - rd) + rd
                    l1 = L * Mv * (~U)
                    V = (
                        l
                        * (2 * rng.random() * positions[i] - positions[a])
                        + l1 * (positions[b] - positions[a])
                        + (1 - Rnorm[i]) * f * u1.astype(float) * rng.random(dim) * (ub - lb)
                    )
                else:
                    U2 = rng.random() > rng.random()
                    V = rng.random(dim) * L * (positions[a] - positions[i]) + (
                        (1 - Rnorm[i])
                        * f
                        * float(U2)
                        * rng.random(dim)
                        * (rng.random() * ub - lb)
                    )

                f = 1 if rng.random() < 0.5 else -1
                positions[i] = (
                    (positions[i] + V * f)
                    + (Fg[i] + abs(rng.normal())) * U.astype(float) * (sun_pos - positions[i])
                )

            if rng.random() < rng.random():
                for j in range(dim):
                    if positions[i, j] > ub[j] or positions[i, j] < lb[j]:
                        positions[i, j] = lb[j] + rng.random() * (ub[j] - lb[j])
            else:
                positions[i] = np.clip(positions[i], lb, ub)

            result = objective_function(positions[i], data_path=data_path, device=device)
            if result.loss_metric < pl_fit[i]:
                pl_fit[i] = result.loss_metric
                predictions[i] = result.predictions
                nets[i] = result.model
                infos[i] = {
                    "history": result.history,
                    "metrics": result.metrics,
                    "position": positions[i].copy(),
                }

                if pl_fit[i] < sun_score:
                    sun_score = pl_fit[i]
                    sun_pos = positions[i].copy()
                    best_prediction = result.predictions
                    best_net = result.model
                    best_info = infos[i]
            else:
                positions[i] = original_pos

            t += 1
            if t >= t_max:
                break
        if t >= t_max:
            break

    koa_curve = np.sort(pl_fit)[::-1]

    best_position = sun_pos.copy()
    if best_position.size:
        best_vec = np.zeros_like(best_position)
        for idx in range(best_position.size):
            if idx == 0:
                best_vec[idx] = best_position[idx]
            else:
                best_vec[idx] = round(best_position[idx])
        best_position = best_vec

    return KOAResult(
        best_score=sun_score,
        best_position=best_position,
        curve=koa_curve,
        best_predictions=best_prediction,
        best_model=best_net,
        best_metrics=best_info,
    )

