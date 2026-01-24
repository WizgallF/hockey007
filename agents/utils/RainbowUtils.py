import numpy as np
import random
from collections import namedtuple, deque


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """
        Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        ----------
        Parameters
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """Value of the schedule at time t"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class ReplayBuffer(object):
    def __init__(self, size, gamma, n_step, prioritized=False, alpha=0.6, beta=0.4, eps=1e-6):
        """
        Create Replay buffer.
        ----------
        Parameters
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        prioritized: bool
            whether to use prioritized sampling
        alpha: float
            priority exponent
        beta: float
            importance-sampling exponent
        eps: float
            small constant to avoid zero priority
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._gamma = gamma
        self._n_step = n_step
        self._prioritized = prioritized
        self._alpha = alpha
        self._beta = beta
        self._eps = eps
        self._max_priority = 1.0
        self._priorities = np.zeros((size,), dtype=np.float32) if prioritized else None

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """ 
        Add a transition to replay memory. 
        Parameters
        ----------
        obs_t: 
            State s_t
        action: 
            Action a_t taken in s_t
        reward: 
            Received reward r_t
        obs_tp1: 
            Follow-up state s_{t+1}
        done: bool
            Whether episode has terminated at s_{t+1}
        """

        
        obs_t = np.ascontiguousarray(obs_t)
        obs_tp1 = np.ascontiguousarray(obs_tp1)

        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        if self._prioritized:
            self._priorities[self._next_idx] = self._max_priority

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data

            returns = 0
            gamma_step = 1
            for step in range(self._n_step):
                if (i + step) % len(self._storage) < (i + step):
                    done = True
                    break

                transition = self._storage[i + step]
                _, _, reward, obs_tp1, done = transition

                returns += reward * gamma_step

                if done:
                    break
                gamma_step *= self._gamma

            # decode back to float32 for training
            obses_t.append(obs_t)
            actions.append(np.asarray(action))
            rewards.append(returns)
            obses_tp1.append(obs_tp1)
            dones.append(done)

        return np.squeeze(np.asarray(obses_t), axis=1), np.asarray(actions), np.asarray(rewards, dtype=np.float32), np.squeeze(np.asarray(obses_tp1), axis=1), np.asarray(dones, dtype=np.float32)

    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if self._prioritized:
            probs = self._get_probs()
            idxes = np.random.choice(len(self._storage), batch_size, p=probs)
            weights = (len(self._storage) * probs[idxes]) ** (-self._beta)
            weights /= weights.max() + 1e-8
        else:
            idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
            weights = np.ones(len(idxes), dtype=np.float32)

        batch = self._encode_sample(idxes)
        return (*batch, weights.astype(np.float32), np.array(idxes, dtype=np.int64))

    def _get_probs(self):
        scaled = np.power(self._priorities[:len(self._storage)], self._alpha)
        total = scaled.sum()
        if total == 0:
            return np.ones(len(self._storage)) / len(self._storage)
        return scaled / total

    def update_priorities(self, idxes, priorities):
        """Update priorities of the training samples
        Parameters
        ----------
        idxes: list of int
            the indexes to update
        priorities: list of floats
            priorities based on the samples squared error between prediction and target
        """
        # priorities are updated based on their loss
        if not self._prioritized:
            return
        for idx, priority in zip(idxes, priorities):
            p = float(abs(priority) + self._eps)
            self._priorities[idx] = p
            if p > self._max_priority:
                self._max_priority = p

    def set_beta(self, beta):
        """Update the beta value 
        Parameters
        ----------
        beta: float
            value to set beta
        """
        if self._prioritized:
            self._beta = beta