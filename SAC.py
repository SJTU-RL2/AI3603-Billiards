from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from agent import Agent, EPS
from stable_baselines3 import SAC as SB3SAC
from stable_baselines3.common.base_class import BaseAlgorithm


class SacAgent(Agent):
    """Stable Baselines3 SAC agent wrapper using the pool observation encoding."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        policy: str = "MlpPolicy",
        deterministic_eval: bool = True,
        training: bool = False,
        **sac_kwargs,
    ) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.policy_name = policy
        self.deterministic_eval = deterministic_eval
        self.training = training
        self.model: Optional[BaseAlgorithm] = None
        self.sac_kwargs = sac_kwargs

        self.action_low = np.full(5, -1.0, dtype=np.float32)
        self.action_high = np.full(5, 1.0, dtype=np.float32)
        self.physical_action_low = np.asarray([0.5, 0.0, 0.0, -0.5, -0.5], dtype=np.float32)
        self.physical_action_high = np.asarray([8.0, 360.0, 90.0, 0.5, 0.5], dtype=np.float32)
        self._action_span = self.physical_action_high - self.physical_action_low

        if model_path:
            self.load(model_path)

    def attach_model(self, model: BaseAlgorithm) -> None:
        self.model = model

    def set_training(self, training: bool) -> None:
        self.training = training

    def decision(
        self,
        balls=None,
        my_targets=None,
        table=None,
        deterministic: Optional[bool] = None,
    ) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("SAC model is not loaded. Call build_model() or load() first.")
        if balls is None or my_targets is None or table is None:
            return self._random_action()

        observation = self._encode_observation(balls, my_targets, table)
        obs = observation.reshape(1, -1)
        use_det = self.deterministic_eval if deterministic is None else deterministic
        action, _ = self.model.predict(obs, deterministic=use_det)
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action[0]
        return self._format_action(np.asarray(action, dtype=np.float32))

    def build_model(self, env, **override_kwargs) -> BaseAlgorithm:
        kwargs = self.sac_kwargs.copy()
        kwargs.update(override_kwargs)
        self.model = SB3SAC(self.policy_name, env, device=self.device, **kwargs)
        return self.model

    def learn(self, total_timesteps: int, **kwargs) -> None:
        if self.model is None:
            raise RuntimeError("Call build_model() or load() before learn().")
        self.training = True
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        self.training = False

    def save(self, filepath: str) -> None:
        if self.model is None:
            raise RuntimeError("SAC model not initialised; cannot save.")
        self.model.save(filepath)

    def load(self, filepath: str, map_location: Optional[str] = None) -> None:
        device = map_location or self.device
        self.model = SB3SAC.load(filepath, device=device)
        self.device = device
        self.training = False

    def _encode_observation(self, balls, my_targets, table) -> np.ndarray:
        ball_ids = ['cue'] + [str(i) for i in range(1, 16)]
        target_set = set(my_targets or [])
        table_l = getattr(table, 'l', 1.0)
        table_w = getattr(table, 'w', 1.0)
        features: List[float] = []
        remaining_targets = 0
        for bid in ball_ids:
            ball = balls.get(bid)
            if ball is None:
                features.extend([0.0] * 10)
                continue
            state = ball.state
            pos = state.rvw[0]
            vel = state.rvw[1]
            pocketed = 1.0 if state.s == 4 else 0.0
            is_target = 1.0 if bid in target_set else 0.0
            if bid in target_set and pocketed == 0.0:
                remaining_targets += 1
            is_black = 1.0 if bid == '8' else 0.0
            features.extend([
                float(pos[0] / (table_l + EPS)),
                float(pos[1] / (table_w + EPS)),
                float(pos[2]),
                float(vel[0] / 5.0),
                float(vel[1] / 5.0),
                float(vel[2] / 5.0),
                float(state.s / 4.0),
                pocketed,
                is_target,
                is_black,
            ])
        target_total = max(1, len(target_set))
        features.extend([
            len(target_set) / 7.0,
            remaining_targets / target_total,
            1.0 if target_set == {'8'} else 0.0,
            1.0,
        ])
        return np.asarray(features, dtype=np.float32)

    def _format_action(self, action: np.ndarray) -> Dict[str, float]:
        clipped = np.clip(action, self.action_low, self.action_high)
        scaled = 0.5 * (clipped + 1.0)
        physical = self.physical_action_low + scaled * self._action_span
        return {
            'V0': float(physical[0]),
            'phi': float(physical[1] % 360.0),
            'theta': float(physical[2]),
            'a': float(np.clip(physical[3], -0.5, 0.5)),
            'b': float(np.clip(physical[4], -0.5, 0.5)),
        }
