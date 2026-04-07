from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ACTION_NAMES = {
    0: "rest",
    1: "passive",
    2: "active",
    3: "strength",
    4: "functional",
}

ACTION_MIN_RECOVERY = {
    0: 0.0,
    1: 0.0,
    2: 0.30,
    3: 0.55,
    4: 0.75,
}


@dataclass(frozen=True)
class PatientProfile:
    name: str
    recovery_gain: float
    pain_sensitivity: float
    fatigue_sensitivity: float
    motivation_drift: float


PATIENT_PROFILES: Dict[str, PatientProfile] = {
    "balanced": PatientProfile("balanced", recovery_gain=1.00, pain_sensitivity=1.00, fatigue_sensitivity=1.00, motivation_drift=0.00),
    "pain_sensitive": PatientProfile("pain_sensitive", recovery_gain=0.92, pain_sensitivity=1.35, fatigue_sensitivity=1.05, motivation_drift=-0.01),
    "low_stamina": PatientProfile("low_stamina", recovery_gain=0.90, pain_sensitivity=1.00, fatigue_sensitivity=1.35, motivation_drift=-0.01),
    "high_motivation": PatientProfile("high_motivation", recovery_gain=1.08, pain_sensitivity=0.90, fatigue_sensitivity=0.90, motivation_drift=0.02),
}


class RehabEnv(gym.Env):
    """Gymnasium rehab environment for 30-day planning."""

    metadata = {"render_modes": []}

    def __init__(self, profile_name: str = "balanced", episode_days: int = 30, seed: int = 42):
        super().__init__()
        self.episode_days = episode_days
        self.rng = np.random.default_rng(seed)
        self.profile_name = profile_name
        self.profile = PATIENT_PROFILES[profile_name]
        self.day = 0
        self.state = np.zeros(4, dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

    def set_profile(self, profile_name: str) -> None:
        self.profile_name = profile_name
        self.profile = PATIENT_PROFILES[profile_name]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        profile_name: str | None = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if options and profile_name is None:
            profile_name = options.get("profile_name")
        if profile_name is not None:
            self.set_profile(profile_name)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.day = 0
        self.state = np.array([
            self.rng.uniform(0.20, 0.35),  # recovery
            self.rng.uniform(0.28, 0.45),  # pain
            self.rng.uniform(0.22, 0.40),  # fatigue
            self.rng.uniform(0.50, 0.75),  # motivation
        ], dtype=np.float32)
        return self._obs(), {"profile": self.profile_name}

    def _obs(self) -> np.ndarray:
        day_normalized = self.day / float(self.episode_days - 1)
        return np.concatenate([self.state, np.array([day_normalized], dtype=np.float32)])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        recovery, pain, fatigue, motivation = [float(v) for v in self.state]
        p = self.profile

        if recovery < ACTION_MIN_RECOVERY[action]:
            action = max(0, action - 1)

        training_load = [0.0, 0.2, 0.45, 0.65, 0.82][action]
        recovery_boost = [0.02, 0.04, 0.07, 0.10, 0.12][action] * p.recovery_gain
        pain_delta = [-0.03, -0.01, 0.02, 0.05, 0.08][action] * p.pain_sensitivity
        fatigue_delta = [-0.05, -0.01, 0.03, 0.07, 0.10][action] * p.fatigue_sensitivity

        readiness = max(0.0, 1.0 - abs(training_load - recovery))
        recovery += recovery_boost * (0.6 + 0.8 * readiness)
        pain += pain_delta + 0.03 * fatigue + self.rng.normal(0.0, 0.01)
        fatigue += fatigue_delta + 0.02 * training_load + self.rng.normal(0.0, 0.012)
        motivation += p.motivation_drift + 0.015 * (1.0 - pain) - 0.01 * fatigue + self.rng.normal(0.0, 0.01)

        if pain > 0.75:
            recovery -= 0.03 + 0.04 * (pain - 0.75)
        if fatigue > 0.80:
            recovery -= 0.02 + 0.05 * (fatigue - 0.80)

        recovery = float(np.clip(recovery, 0.0, 1.0))
        pain = float(np.clip(pain, 0.0, 1.0))
        fatigue = float(np.clip(fatigue, 0.0, 1.0))
        motivation = float(np.clip(motivation, 0.0, 1.0))
        self.state = np.array([recovery, pain, fatigue, motivation], dtype=np.float32)

        self.day += 1
        done = self.day >= self.episode_days

        reward = (
            2.8 * recovery
            - 1.9 * pain
            - 1.6 * fatigue
            + 0.6 * motivation
            - 0.08 * action
        )
        if done:
            reward += 2.0 * recovery - 1.0 * pain
            if recovery >= 0.80 and pain <= 0.35:
                reward += 4.0

        info = {
            "recovery": recovery,
            "pain": pain,
            "fatigue": fatigue,
            "motivation": motivation,
            "day": self.day,
            "action": action,
        }
        truncated = False
        return self._obs(), float(reward), done, truncated, info


def rule_based_policy(obs: np.ndarray, profile: str | None = None) -> int:
    recovery, pain, fatigue, motivation, day_normalized = [float(x) for x in obs]

    if pain >= 0.70 or fatigue >= 0.78:
        return 0

    if recovery < 0.30:
        return 0 if pain > 0.50 or fatigue > 0.55 else 1

    if recovery < 0.55:
        if pain > 0.55 or fatigue > 0.60:
            return 1
        return 2

    if recovery < 0.75:
        if pain > 0.50 or fatigue > 0.58:
            return 1 if day_normalized < 0.50 else 2
        return 3 if motivation > 0.35 else 2

    if pain < 0.35 and fatigue < 0.45 and motivation > 0.45:
        return 4

    if pain < 0.50 and fatigue < 0.55:
        return 3

    return 2


BINS = {
    "recovery": np.linspace(0.0, 1.0, 11),
    "pain": np.linspace(0.0, 1.0, 11),
    "fatigue": np.linspace(0.0, 1.0, 11),
    "motivation": np.linspace(0.0, 1.0, 9),
    "day": np.linspace(0.0, 1.0, 7),
}


def digitize(obs: np.ndarray) -> Tuple[int, int, int, int, int]:
    recovery, pain, fatigue, motivation, day_norm = obs
    return (
        int(np.digitize(recovery, BINS["recovery"]) - 1),
        int(np.digitize(pain, BINS["pain"]) - 1),
        int(np.digitize(fatigue, BINS["fatigue"]) - 1),
        int(np.digitize(motivation, BINS["motivation"]) - 1),
        int(np.digitize(day_norm, BINS["day"]) - 1),
    )


def train_q_learning(
    env: RehabEnv,
    episodes: int = 5000,
    alpha: float = 0.16,
    gamma: float = 0.98,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.999,
    seed: int = 42,
) -> Dict[Tuple[int, int, int, int, int], np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)

    q_table: Dict[Tuple[int, int, int, int, int], np.ndarray] = {}
    profile_names = list(PATIENT_PROFILES)
    epsilon = eps_start
    rewards = []

    for ep in range(episodes):
        profile_name = profile_names[ep % len(profile_names)]
        obs, _ = env.reset(profile_name=profile_name, seed=seed + ep)
        state = digitize(obs)
        total_reward = 0.0

        done = False
        while not done:
            if state not in q_table:
                q_table[state] = np.zeros(5, dtype=np.float32)

            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                action = int(np.argmax(q_table[state]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = digitize(next_obs)
            if next_state not in q_table:
                q_table[next_state] = np.zeros(5, dtype=np.float32)

            best_next = float(np.max(q_table[next_state]))
            td_target = reward + gamma * best_next * (0.0 if done else 1.0)
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            total_reward += reward
            state = next_state

        epsilon = max(eps_end, epsilon * eps_decay)
        rewards.append(total_reward)

        if (ep + 1) % 500 == 0:
            mean_last = float(np.mean(rewards[-200:]))
            print(f"episode={ep+1:4d} epsilon={epsilon:.3f} avg_reward(last200)={mean_last:.3f}")

    return q_table


def q_policy(obs: np.ndarray, q_table: Dict[Tuple[int, int, int, int, int], np.ndarray]) -> int:
    state = digitize(obs)
    if state not in q_table:
        return rule_based_policy(obs)
    return int(np.argmax(q_table[state]))


def run_policy(
    env: RehabEnv,
    policy_name: str,
    policy_fn,
    episodes_per_profile: int = 25,
    seed: int = 123,
) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    profile_stats: Dict[str, List[dict]] = {name: [] for name in PATIENT_PROFILES}

    for profile_idx, profile_name in enumerate(PATIENT_PROFILES):
        for episode in range(episodes_per_profile):
            obs, _ = env.reset(profile_name=profile_name, seed=seed + 10_000 * profile_idx + episode)
            done = False
            total_reward = 0.0
            final_info = {}

            while not done:
                action = int(policy_fn(obs, profile_name))
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                rows.append(
                    {
                        "profile": profile_name,
                        "episode": episode,
                        "day": info["day"],
                        "action": info["action"],
                        "action_name": ACTION_NAMES[info["action"]],
                        "reward": reward,
                        "recovery": info["recovery"],
                        "pain": info["pain"],
                        "fatigue": info["fatigue"],
                        "motivation": info["motivation"],
                    }
                )

                obs = next_obs
                final_info = info

            profile_stats[profile_name].append(
                {
                    "total_reward": total_reward,
                    "final_recovery": final_info.get("recovery", 0.0),
                    "final_pain": final_info.get("pain", 0.0),
                    "success": 1.0 if (final_info.get("recovery", 0.0) >= 0.80 and final_info.get("pain", 0.0) <= 0.35) else 0.0,
                }
            )

    all_eps = [x for vals in profile_stats.values() for x in vals]
    summary = {
        "policy": policy_name,
        "avg_total_reward": float(np.mean([e["total_reward"] for e in all_eps])),
        "avg_final_recovery": float(np.mean([e["final_recovery"] for e in all_eps])),
        "avg_final_pain": float(np.mean([e["final_pain"] for e in all_eps])),
        "success_rate": float(np.mean([e["success"] for e in all_eps])),
        "per_profile": {
            pname: {
                "avg_total_reward": float(np.mean([e["total_reward"] for e in eps])),
                "avg_final_recovery": float(np.mean([e["final_recovery"] for e in eps])),
                "avg_final_pain": float(np.mean([e["final_pain"] for e in eps])),
                "success_rate": float(np.mean([e["success"] for e in eps])),
            }
            for pname, eps in profile_stats.items()
        },
    }
    return rows, summary


def write_predictions_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "profile",
        "episode",
        "day",
        "action",
        "action_name",
        "reward",
        "recovery",
        "pain",
        "fatigue",
        "motivation",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(title: str, summary: dict) -> None:
    print(f"\n=== {title} ===")
    print(
        f"avg_reward={summary['avg_total_reward']:.3f} | "
        f"avg_final_recovery={summary['avg_final_recovery']:.3f} | "
        f"avg_final_pain={summary['avg_final_pain']:.3f} | "
        f"success_rate={summary['success_rate']:.1%}"
    )
    for profile, stats in summary["per_profile"].items():
        print(
            f"- {profile:14s} reward={stats['avg_total_reward']:.3f} "
            f"recovery={stats['avg_final_recovery']:.3f} "
            f"pain={stats['avg_final_pain']:.3f} "
            f"success={stats['success_rate']:.1%}"
        )


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    env = RehabEnv(seed=42)

    print("Training RL agent (tabular Q-learning)...")
    q_table = train_q_learning(env, episodes=5000, seed=42)

    rule_rows, rule_summary = run_policy(
        env,
        policy_name="rule_based",
        policy_fn=lambda obs, profile: rule_based_policy(obs, profile),
        episodes_per_profile=25,
        seed=2026,
    )
    write_predictions_csv(output_dir / "rulebased_predictions.csv", rule_rows)

    rl_rows, rl_summary = run_policy(
        env,
        policy_name="q_learning",
        policy_fn=lambda obs, profile: q_policy(obs, q_table),
        episodes_per_profile=25,
        seed=2026,
    )
    write_predictions_csv(output_dir / "rl_predictions.csv", rl_rows)

    print_summary("Rule-based baseline", rule_summary)
    print_summary("RL agent (Q-learning)", rl_summary)

    delta = rl_summary["avg_total_reward"] - rule_summary["avg_total_reward"]
    print(f"\nRL improvement in avg total reward: {delta:+.3f}")


if __name__ == "__main__":
    main()
