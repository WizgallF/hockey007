import numpy as np
import random
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Union


@dataclass
class Episode:
    """Stores one episode as numpy arrays and (optionally) per-start priorities."""
    eid: int
    obs: np.ndarray            # shape (T+1, *obs_shape)
    actions: np.ndarray        # shape (T, *act_shape) or (T,)
    rewards: np.ndarray        # shape (T,)
    dones: np.ndarray          # shape (T,)
    start_priorities: Optional[np.ndarray] = None  # shape (num_starts,)


class TDMPC2ReplayBuffer:
    """
    Episode-based sequence replay buffer for TD-MPC2 style updates.

    Key properties:
    - Stores transitions grouped by episode.
    - Samples contiguous segments of length `horizon`:
        obs[t:t+H+1], actions[t:t+H], rewards[t:t+H], dones[t:t+H]
    - Never crosses episode boundaries.
    - Optional prioritized sampling over segment start positions.
    """

    def __init__(
        self,
        capacity_steps: int,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        capacity_steps : int
            Maximum number of environment steps (transitions) stored across episodes.
        prioritized : bool
            If True, use PER over segment start positions.
        alpha : float
            Priority exponent.
        beta : float
            Importance sampling exponent.
        eps : float
            Small constant to avoid zero priorities.
        seed : Optional[int]
            RNG seed.
        """
        self.capacity_steps = int(capacity_steps)
        self.prioritized = bool(prioritized)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

        self._rng = np.random.RandomState(seed) if seed is not None else np.random

        self._episodes: Deque[Episode] = deque()
        self._episode_by_id: Dict[int, Episode] = {}

        self._cur_obs: List[np.ndarray] = []
        self._cur_actions: List[np.ndarray] = []
        self._cur_rewards: List[float] = []
        self._cur_dones: List[bool] = []

        self._next_eid = 0
        self._total_steps = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        """Total number of transitions stored (sum over episodes)."""
        return self._total_steps

    # ---------------------------
    # Data insertion
    # ---------------------------

    def add(self, state, action, reward, next_state, terminated: bool):
        """
        Add one transition (obs_t, action, reward, obs_{t+1}, done).

        Notes:
        - We store observations as an episode list of length T+1:
            obs[0]=s0, obs[1]=s1, ..., obs[T]=sT
        - We store actions/rewards/dones of length T.
        """
        state = np.ascontiguousarray(state)
        next_state = np.ascontiguousarray(next_state)

        # Start new episode storage if needed
        if len(self._cur_obs) == 0:
            self._cur_obs.append(state)
        else:
            # Ensure obs_t matches last stored obs (optional sanity check)
            # If you have wrappers that change obs shapes, you may disable this.
            pass

        self._cur_actions.append(np.asarray(action))
        self._cur_rewards.append(float(reward))
        self._cur_dones.append(bool(terminated))
        self._cur_obs.append(next_state)

        if terminated:
            self._finalize_current_episode()

    def end_episode(self):
        """
        Call this if your environment can truncate episodes without done=True,
        e.g., time-limit truncation where you still want to seal the episode.
        """
        if len(self._cur_actions) > 0:
            # Mark the final transition as done if it wasn't already.
            if not self._cur_dones[-1]:
                self._cur_dones[-1] = True
            self._finalize_current_episode()

    def _finalize_current_episode(self):
        """Package current episode lists into numpy arrays and push to deque."""
        T = len(self._cur_actions)
        if T == 0:
            # nothing to finalize
            self._cur_obs.clear()
            return

        obs = np.stack(self._cur_obs, axis=0)  # (T+1, ...)
        actions = np.stack(self._cur_actions, axis=0) if np.asarray(self._cur_actions[0]).ndim > 0 else np.asarray(self._cur_actions)
        rewards = np.asarray(self._cur_rewards, dtype=np.float32)
        dones = np.asarray(self._cur_dones, dtype=np.bool_)

        ep = Episode(
            eid=self._next_eid,
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            start_priorities=None,
        )
        self._next_eid += 1

        # Store episode
        self._episodes.append(ep)
        self._episode_by_id[ep.eid] = ep

        # Update total step count
        self._total_steps += T

        # Reset current episode buffers
        self._cur_obs = []
        self._cur_actions = []
        self._cur_rewards = []
        self._cur_dones = []

        # Enforce capacity by evicting oldest episodes
        self._evict_if_needed()

    def _evict_if_needed(self):
        """Pop oldest episodes until total steps <= capacity_steps."""
        while self._total_steps > self.capacity_steps and len(self._episodes) > 0:
            old = self._episodes.popleft()
            T = len(old.rewards)
            self._total_steps -= T
            self._episode_by_id.pop(old.eid, None)

    # ---------------------------
    # Sampling (segments)
    # ---------------------------

    def sample_sequences(
        self,
        batch_size: int,
        horizon: int,
        allow_terminal_inside: bool = True,
    ):
        """
        Sample a batch of contiguous segments of length `horizon`.

        Returns
        -------
        obs : np.ndarray
            (B, H+1, *obs_shape)
        actions : np.ndarray
            (B, H, *act_shape) or (B, H) if discrete
        rewards : np.ndarray
            (B, H)
        dones : np.ndarray
            (B, H) float32 mask (1.0 if done else 0.0)
        weights : np.ndarray
            (B,) importance sampling weights (ones if not prioritized)
        idxes : list
            list of (episode_id, start_index) pairs for priority updates
        """
        assert horizon >= 1
        assert batch_size >= 1

        # Eligible episodes must have at least H transitions
        eligible = [ep for ep in self._episodes if len(ep.rewards) >= horizon]
        if len(eligible) == 0:
            raise RuntimeError(
                f"No eligible episodes for horizon={horizon}. "
                f"Need at least one episode with >= {horizon} transitions."
            )

        if self.prioritized:
            self._ensure_priorities_for_horizon(eligible, horizon)
            pairs, weights = self._prioritized_sample_pairs(eligible, batch_size)
        else:
            pairs = []
            for _ in range(batch_size):
                ep = random.choice(eligible)
                start = self._uniform_valid_start(ep, horizon, allow_terminal_inside)
                pairs.append((ep.eid, start))
            weights = np.ones((batch_size,), dtype=np.float32)

        # Gather segments
        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []
        for eid, start in pairs:
            ep = self._episode_by_id[eid]

            o = ep.obs[start : start + horizon + 1]                           # (H+1, ...)
            a = ep.actions[start : start + horizon]                           # (H, ...)
            r = ep.rewards[start : start + horizon].astype(np.float32)        # (H,)
            d = ep.dones[start : start + horizon].astype(np.float32)          # (H,)

            obs_batch.append(o)
            act_batch.append(a)
            rew_batch.append(r)
            done_batch.append(d)

        obs_batch = np.stack(obs_batch, axis=0)
        act_batch = np.stack(act_batch, axis=0)
        rew_batch = np.stack(rew_batch, axis=0)
        done_batch = np.stack(done_batch, axis=0)

        return obs_batch, act_batch, rew_batch, done_batch, weights.astype(np.float32), pairs

    def _uniform_valid_start(self, ep: Episode, horizon: int, allow_terminal_inside: bool) -> int:
        """Uniformly sample a valid start index for a length-H segment."""
        T = len(ep.rewards)
        max_start = T - horizon  # inclusive
        if max_start < 0:
            raise RuntimeError("Episode too short for horizon.")

        if allow_terminal_inside:
            return random.randint(0, max_start)

        # Require no terminal in first H-1 steps; terminal at final step is okay.
        candidates = []
        for s in range(max_start + 1):
            if not ep.dones[s : s + horizon - 1].any():
                candidates.append(s)
        if not candidates:
            return random.randint(0, max_start)
        return random.choice(candidates)

    # ---------------------------
    # Prioritized sampling over segment starts
    # ---------------------------

    def _ensure_priorities_for_horizon(self, eligible: List[Episode], horizon: int):
        """
        Initialize or resize per-episode start priorities for current horizon.

        For episode length T, number of valid starts = T - H + 1.
        """
        for ep in eligible:
            T = len(ep.rewards)
            n_starts = T - horizon + 1
            if n_starts <= 0:
                continue
            if ep.start_priorities is None or len(ep.start_priorities) != n_starts:
                ep.start_priorities = np.full((n_starts,), self._max_priority, dtype=np.float32)

    def _prioritized_sample_pairs(self, eligible: List[Episode], batch_size: int):
        """
        Sample (episode_id, start) using per-start priorities across all eligible episodes.
        """
        flat_refs: List[Tuple[int, int]] = []
        flat_pieces: List[np.ndarray] = []

        for ep in eligible:
            if ep.start_priorities is None:
                continue
            p = np.power(ep.start_priorities + self.eps, self.alpha)
            if p.sum() <= 0:
                continue
            # add refs for each start
            for s in range(len(p)):
                flat_refs.append((ep.eid, s))
            flat_pieces.append(p)

        if len(flat_refs) == 0:
            # fallback uniform
            pairs = []
            for _ in range(batch_size):
                ep = random.choice(eligible)
                start = random.randint(0, len(ep.rewards) - horizon)
                pairs.append((ep.eid, start))
            weights = np.ones((batch_size,), dtype=np.float32)
            return pairs, weights

        flat_p = np.concatenate(flat_pieces, axis=0).astype(np.float64)
        flat_p /= (flat_p.sum() + 1e-12)

        chosen = self._rng.choice(len(flat_refs), size=batch_size, replace=True, p=flat_p)
        pairs = [flat_refs[i] for i in chosen]

        probs = flat_p[chosen]
        weights = (len(flat_refs) * probs) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)
        return pairs, weights.astype(np.float32)

    def update_priorities(self, idxes: List[Tuple[int, int]], priorities: Union[np.ndarray, List[float]]):
        """
        Update priorities for sampled segment starts.

        idxes : list of (episode_id, start_index)
        priorities : array-like of shape (B,)
        """
        if not self.prioritized:
            return

        priorities = np.asarray(priorities, dtype=np.float32)
        assert len(idxes) == len(priorities)

        for (eid, start), pr in zip(idxes, priorities):
            ep = self._episode_by_id.get(eid, None)
            if ep is None or ep.start_priorities is None:
                continue
            p = float(abs(pr) + self.eps)
            if 0 <= start < len(ep.start_priorities):
                ep.start_priorities[start] = p
                if p > self._max_priority:
                    self._max_priority = p

    def set_beta(self, beta: float):
        """Set PER importance-sampling exponent beta."""
        self.beta = float(beta)

