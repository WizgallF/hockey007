import numpy as np
import torch
import time
from agents.AgentBaseclass import Agent 
from agents.RainbowAgent import RainbowAgent
from agents.DDPGAgent import DDPGAgent
from agents.TDMPC2Agent import TDMPC2Agent
import os
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import count
from Wrapper import Envwrapper, DiscreteActionWrapperHockey
import hockey.hockey_env as h_env
from copy import deepcopy
import random
import yaml
from gymnasium import spaces
from collections import Counter


class Training():
    def __init__(
        self,
        agent = None,
        env = None,
        base_dir = "experiments",
        save_intermediate_agents: bool = False,
        verbose=False
        ):
        
        self.statistics = {
            "ep_rew": [0.0],
            "mv_avg_rew": [],
            "tr_loss": [],
            "min_q": [],
            "mean_q": [],
            "max_q": []
        }

        self.agent = agent
        self.env = env
        self.base_dir = base_dir
        self.save_intermediate_agents = save_intermediate_agents
        self.verbose = verbose
        self.mavg_window_size = int(self.agent.NUM_EPISODES / 100)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def train(self):
        
        torch.set_default_dtype(torch.float32)

        # print hyperparameter settings to console 
        if self.verbose:
            self.agent.print_config()

        # Create experiment folder and save config
        self.experiment_path = self.agent.save_experiment_config(self.base_dir)


        start = time.time()
        best_mv_avg_reward = float('-inf')

        is_vector_env = hasattr(self.env, "num_envs")
        print(is_vector_env)
        if not is_vector_env:
            for i_episode in range(self.agent.NUM_EPISODES):
                self.agent.cur_episode = i_episode

                # --------- init environment -----------
                state, info = self.env.reset()
                
                for t in count():

                    # ------ act ------
                    action = self.agent.act(self.env, state, i_episode, self.statistics)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)


                    # ------ observe ------
                    self.agent.observe(state, action, reward, next_state, terminated)
                    self.statistics["ep_rew"][-1] += float(reward) 
                    

                    # ------ move to next state ------
                    state = next_state


                    # ------ update ------
                    if i_episode >= self.agent.START_TRAINING:
                        self.agent.update(self.statistics)

                    # ------ terminate episode ------
                    done = terminated or truncated
                    if done:
                        self.statistics["ep_rew"].append(0)
                        break
                


                # ------ save best performing agent ------
                n = len(self.statistics["ep_rew"]) 
                if n > self.mavg_window_size + 1:
                    mv_avg_reward = np.mean(self.statistics["ep_rew"][-self.mavg_window_size:-1])
                    self.statistics["mv_avg_rew"].append(mv_avg_reward)
                    if mv_avg_reward > best_mv_avg_reward:
                        self.agent.save_dict(self.experiment_path)
                        best_mv_avg_reward = mv_avg_reward
                        if self.verbose:
                            print(f"Saved model with moving average reward: {mv_avg_reward}")


                # ------ print to console -----
                if self.verbose and i_episode % 5 == 0:
                    end = time.time()
                    print(f"\n** after {i_episode} th episode - {end - start:.5f} sec passed**\n")
        else:
            num_envs = int(self.env.num_envs)
            agent_type = type(self.agent).__name__
            assert agent_type in {"DDPGAgent", "TDMPC2Agent", "RainbowAgent"}, (
                "Parallel training is only supported for DDPGAgent and TDMPC2Agent."
            )
            is_ddpg_vectorized = agent_type == "DDPGAgent"
            ep_rew_per_env = np.zeros(num_envs, dtype=np.float32)
            state, info = self.env.reset()
            episodes_finished = 0

            while episodes_finished < self.agent.NUM_EPISODES:
                self.agent.cur_episode = episodes_finished

                # ------ act ------
                if is_ddpg_vectorized:
                    action = self.agent.act(self.env, state, episodes_finished, self.statistics)
                else:
                    action = np.asarray(
                        [
                            self.agent.act(self.env, state[env_idx], episodes_finished, self.statistics)
                            for env_idx in range(num_envs)
                        ]
                    )

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = np.logical_or(terminated, truncated)

                # ------ observe ------
                if is_ddpg_vectorized:
                    self.agent.observe(state, action, reward, next_state, terminated)
                else:
                    for env_idx in range(num_envs):
                        self.agent.observe(
                            state[env_idx],
                            action[env_idx],
                            float(reward[env_idx]),
                            next_state[env_idx],
                            bool(done[env_idx]),
                        )
                ep_rew_per_env += reward.astype(np.float32)

                # ------ move to next state ------
                state = next_state

                # ------ update ------
                if episodes_finished >= self.agent.START_TRAINING:
                    self.agent.update(self.statistics)

                # ------ terminate episodes ------
                if np.any(done):
                    done_indices = np.where(done)[0]
                    for env_idx in done_indices:
                        self.statistics["ep_rew"].append(float(ep_rew_per_env[env_idx]))
                        ep_rew_per_env[env_idx] = 0.0

                    # Reset only completed environments if supported
                    if hasattr(self.env, "reset_done"):
                        reset_obs, reset_info = self.env.reset_done()
                        if isinstance(reset_obs, np.ndarray) and reset_obs.shape == state.shape:
                            state = np.where(done[:, None], reset_obs, state)
                        else:
                            for idx, env_idx in enumerate(done_indices):
                                state[env_idx] = reset_obs[idx]
                    else:
                        state, info = self.env.reset()

                # ------ save best performing agent ------
                n = len(self.statistics["ep_rew"]) 
                episodes_finished = n - 1
                if n > self.mavg_window_size + 1:
                    mv_avg_reward = np.mean(self.statistics["ep_rew"][-self.mavg_window_size:-1])
                    self.statistics["mv_avg_rew"].append(mv_avg_reward)
                    if mv_avg_reward > best_mv_avg_reward:
                        self.agent.save_dict(self.experiment_path)
                        best_mv_avg_reward = mv_avg_reward
                        if self.verbose:
                            print(f"Saved model with moving average reward: {mv_avg_reward}")

                # ------ print to console -----
                if self.verbose and episodes_finished % 5 == 0:
                    end = time.time()
                    print(
                        f"\n** after {episodes_finished} th episode - {end - start:.5f} sec passed**\n"
                    )
            

        self.save_data()

        if type(self.agent) == RainbowAgent:
            self.save_q_values()
    
    def save_data(self, mean_eval_score = 0):

        # ------ create reward plot -------
        mavg_data = np.array(self.statistics["mv_avg_rew"])
        # Create an x-axis that starts at self.mavg_window_size
        x_axis = np.arange(self.mavg_window_size, self.mavg_window_size + len(mavg_data))
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(np.array(self.statistics["ep_rew"]))
        plt.plot(np.array(self.statistics["ep_rew"]), label="Episode Rewards", color="blue", linewidth=1.5)
        plt.plot(x_axis, np.array(self.statistics["mv_avg_rew"]), label="Moving Average Episode Rewards", color="red", linestyle="--", alpha=0.7)
        
        plt.xlabel("Episode Number")
        plt.ylabel("Rewards per Episode")
        plt.title(f"Rewards per Episode over Time. Mean Evaluation Score: {mean_eval_score:.3f}")
        plt.savefig(os.path.join(self.experiment_path, f"episode_rewards-{self.agent.MODEL_IDENTIFIER}.png"), dpi=300)
        plt.close()

        # ------ create loss plot ------
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(np.array(self.statistics["tr_loss"]))
        plt.xlabel("Time Step")
        plt.ylabel("Loss")
        plt.title("Training Loss per Time Step")
        plt.savefig(os.path.join(self.experiment_path, f"training_losses-{self.agent.MODEL_IDENTIFIER}.png"), dpi=300)
        plt.close()

    def save_q_values(self):
        # ------ create q-value plot ------
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(np.array(self.statistics["mean_q"]), label="Mean Q", color="blue", linewidth=1.5)
        plt.plot(np.array(self.statistics["max_q"]), label="Max Q", color="green", linestyle="--", alpha=0.7)
        plt.plot(np.array(self.statistics["min_q"]), label="Min Q", color="red", linestyle="--", alpha=0.7)
        plt.xlabel("Time Step")
        plt.ylabel("Q-values")
        plt.title("Training Loss per Time Step")
        plt.savefig(os.path.join(self.experiment_path, f"q_values-{self.agent.MODEL_IDENTIFIER}.png"), dpi=300)
        plt.close()

    
    def train_self_play(
            self, 
            opponent,
            discrete_actions = False,
            agent_load_path = None,
            population_path = None,
            num_parallel_envs: int = 1):

        torch.set_default_dtype(torch.float32)

        self.original_env = self.env
        self.opponent = self._copy_agent_without_replay_buffer()
        print(type(self.opponent))
        self.best_opponent = self._copy_agent_without_replay_buffer()
        self.population_size = 0
        self.MODEL_IDENTIFIER = self.agent.MODEL_IDENTIFIER
        self.fixed_opponents = getattr(self.agent, "FIXED_OPPONENTS", False)
        self.fixed_opponents_path = population_path
        training_rounds = int(getattr(self.agent, "TRAINING_ROUNDS", 1))
        k_against_strong = int(getattr(self.agent, "K_AGAINST_STRONG", 0))
        use_parallel_mode = int(num_parallel_envs) > 1

        # print hyperparameter settings to console 
        if self.verbose:
            self.agent.print_config()

        # Create experiment folder and save config
        self.experiment_path = self.agent.save_experiment_config(self.base_dir)


        # Add random agent to population for self play
        population_path = os.path.join(self.experiment_path, "agent_population")

        # init winrate against fixed opponent pool
        if self.fixed_opponents:
            files = [f for f in os.listdir(self.fixed_opponents_path) if os.path.isfile(os.path.join(self.fixed_opponents_path, f)) and f.split(".")[1] == "pth"]
            N = len(files)
            self.fixed_opponents_labels = files
            self.fixed_opponents_labels.extend(["weak opp", "strong opp"])
            self.last_winrate_vs_pool = np.zeros(shape=(N + 2,))
            self.all_winrates_vs_pool = []
    
        self.save_to_population(population_path, i_training_round=0)
        

        start = time.time()
        best_mv_avg_reward = float('-inf')
        self.agent_against_basic_opp = []
        self.agent_against_strong_opp = []
        self.last_selected_opponent_info = "none"

        if self.verbose:
            print(
                f"[SelfPlay] start | rounds={training_rounds} | episodes_per_round={self.agent.NUM_EPISODES} "
                f"| fixed_opponents={self.fixed_opponents} | warmup_strong_rounds={k_against_strong} "
                f"| parallel_envs={max(1, int(num_parallel_envs))}"
            )


        for i_training_round in range(training_rounds):
            parallel_round_stats = None
            if self.verbose:
                print(
                    f"\n[SelfPlay][Round {i_training_round + 1}/{training_rounds}] "
                    f"population_size={self.population_size}"
                )

            # Wrap environment with Player 2

            if use_parallel_mode:
                schedule = self._build_parallel_opponent_schedule(
                    num_envs=int(num_parallel_envs),
                    population_path=population_path,
                )
                if self.verbose:
                    print(
                        f"[SelfPlay][Round {i_training_round + 1}] mode=parallel_pool "
                        f"| parallel_envs={len(schedule)} | {self._format_parallel_composition(schedule)}"
                    )
                parallel_round_stats = self._run_self_play_parallel_round(
                    schedule,
                    discrete_actions,
                    i_training_round,
                    start,
                )
                if self.verbose:
                    self._log_parallel_round_summary(i_training_round, parallel_round_stats)
            else:
                # Train first rounds against strong opponent (single-env mode only)
                if not self.fixed_opponents and i_training_round < k_against_strong:
                    self.opponent = h_env.BasicOpponent(weak=False)
                    self.env = Envwrapper(self.original_env, self.opponent, discrete_actions)
                    if self.verbose:
                        print(f"[SelfPlay][Round {i_training_round + 1}] mode=warmup_strong_opponent")
                    self._run_self_play_single_round(i_training_round, start)
                else:
                    self.select_from_population(population_path)
                    self.env = Envwrapper(self.original_env, self.opponent, discrete_actions)
                    if self.verbose:
                        print(
                            f"[SelfPlay][Round {i_training_round + 1}] mode=single_opponent "
                            f"| selected={self.last_selected_opponent_info}"
                        )
                    self._run_self_play_single_round(i_training_round, start)

            winrate_vs_best = self.agent_against_agent_eval(self.agent, self.best_opponent)

            if not self.fixed_opponents:
                winrate_vs_best = self.agent_against_agent_eval(self.agent, self.best_opponent)
                if self.verbose:
                    print(
                        f"[SelfPlay][Round {i_training_round + 1}] end "
                        f"| winrate_vs_best={winrate_vs_best:.3f} | {self._latest_self_play_stats()}"
                    )

                if winrate_vs_best > 0.5:
                    self.best_opponent = self._copy_agent_without_replay_buffer()
                    self.save_to_population(population_path, i_training_round)
                    if self.verbose:
                        print(
                            f"[SelfPlay][Round {i_training_round + 1}] "
                            f"new_best_saved | population_size={self.population_size}"
                        )
            else:
                opponents = self._load_population_opponents(self.fixed_opponents_path)

                winrate_vs_pool = self._agent_against_pool_eval(self.agent, opponents)
                self.all_winrates_vs_pool.append(winrate_vs_pool)

                if np.sum(winrate_vs_pool - self.last_winrate_vs_pool) > 0:
                    self.best_opponent = self._copy_agent_without_replay_buffer()
                    self.save_to_population(population_path, i_training_round)
                    if self.verbose:
                        print(
                            f"[SelfPlay][Round {i_training_round + 1}] "
                            f"new_best_saved | population_size={self.population_size}"
                        )
            
            

        self.save_data()
        self.save_performance_against_basic_opp()

        if self.fixed_opponents:
            self.save_performance_against_fixed_pool()

        if type(self.agent) == RainbowAgent:
            self.save_q_values()

        if not self.fixed_opponents:
            eval_results = self.evaluate_agents(population_path)
        
    def _run_self_play_single_round(
            self,
            i_training_round,
            start):
        
        for i_episode in range(self.agent.NUM_EPISODES):
            self.agent.cur_episode = i_episode

            # --------- init environment -----------
            state, info = self.env.reset()
            
            for t in count():

                # ------ act ------
                action = self.agent.act(self.env, state, i_episode, self.statistics)
                next_state, reward, terminated, truncated, _ = self.env.step(action)


                # ------ observe ------
                self.agent.observe(state, action, reward, next_state, terminated)
                self.statistics["ep_rew"][-1] += float(reward) 
                

                # ------ move to next state ------
                state = next_state


                # ------ update ------
                if i_episode >= self.agent.START_TRAINING:
                    self.agent.update(self.statistics)

                # ------ terminate episode ------
                done = terminated or truncated
                if done:
                    self.statistics["ep_rew"].append(0)
                    break
            


            # TODO: how to determine which agent to save? just save them regularly?
            # ------ save best performing agent ------
            n = len(self.statistics["ep_rew"]) 
            if n > self.mavg_window_size + 1:
                mv_avg_reward = np.mean(self.statistics["ep_rew"][-self.mavg_window_size:-1])
                self.statistics["mv_avg_rew"].append(mv_avg_reward)

                

            # ------ print to console -----
            if self.verbose and (i_episode == 0 or (i_episode + 1) % 100 == 0):
                end = time.time()
                print(
                    f"[SelfPlay][Round {i_training_round + 1}] "
                    f"episode={i_episode + 1}/{self.agent.NUM_EPISODES} "
                    f"| elapsed={end - start:.1f}s | {self._latest_self_play_stats()}"
                )

            if i_episode % 1000 == 0:
                agent_basic_eval = self.agent_against_basicopp_eval(self.agent)
                agent_strong_eval = self.agent_against_basicopp_eval(self.agent, weak=False)
                self.agent_against_basic_opp.append(agent_basic_eval)
                self.agent_against_strong_opp.append(agent_strong_eval)
                if self.verbose:
                    print(
                        f"[SelfPlay][Round {i_training_round + 1}] "
                        f"weak_basicopp_eval@episode={i_episode} | winrate={agent_basic_eval:.3f} "
                        f"strong_basicopp_eval@episode={i_episode} | winrate={agent_strong_eval:.3f}"
                    )
    
    def _load_population_opponents(
            self,
            population_path):
        pool = self._load_population_pool(population_path)
        return [item["agent"] for item in pool]

    def _population_sort_key(self, fname: str) -> tuple[int, int, str]:
        stem = os.path.splitext(fname)[0]
        suffix = stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return (0, int(suffix), fname)
        return (1, 0, fname)

    def _strip_agent_training_state(self, agent):
        for attr in ("replay_buffer", "buffer", "action_noise"):
            if hasattr(agent, attr):
                setattr(agent, attr, None)
        return agent

    def _active_parallel_pool_path(self, population_path: str):
        if self.fixed_opponents:
            return self.fixed_opponents_path, "fixed_pool"
        return population_path, "self_play_pool"

    def _load_population_pool(
            self,
            population_path,
            source: str = "self_play_pool"):
        pool = []
        if not population_path or not os.path.isdir(population_path):
            return pool

        candidate_files = [
            fname for fname in os.listdir(population_path)
            if fname.endswith(".pth") and os.path.isfile(os.path.join(population_path, fname))
        ]
        if not candidate_files:
            return pool

        base_env = h_env.HockeyEnv()
        try:
            base_obs_space = base_env.observation_space
            base_act_space = base_env.action_space

            if hasattr(base_obs_space, "shape") and base_obs_space.shape is not None:
                n_observations = int(base_obs_space.shape[0])
            else:
                state, _ = base_env.reset()
                n_observations = len(state)

            try:
                proxy_env = DiscreteActionWrapperHockey(base_env)
                n_actions = int(proxy_env.action_space.n)
            except Exception:
                n_actions = 8

            if hasattr(base_act_space, "low") and hasattr(base_act_space, "high"):
                low = base_act_space.low
                high = base_act_space.high
                if getattr(base_act_space, "shape", None) and base_act_space.shape[0] >= 4:
                    low = low[:4]
                    high = high[:4]
                single_player_action_space = spaces.Box(
                    low=low,
                    high=high,
                    dtype=base_act_space.dtype,
                )
            else:
                single_player_action_space = base_act_space

            sorted_files = sorted(
                candidate_files,
                key=self._population_sort_key,
            )

            for fname in sorted_files:
                load_path = os.path.join(population_path, fname)
                config_path = os.path.join(population_path, os.path.splitext(fname)[0] + ".yaml")
                if not os.path.isfile(config_path):
                    if self.verbose:
                        print(f"[SelfPlay] missing config for opponent: {config_path}")
                    continue

                try:
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f) or {}
                except Exception as exc:
                    if self.verbose:
                        print(f"[SelfPlay] failed to load config {config_path}: {exc}")
                    continue

                model_id = str(config.get("MODEL_IDENTIFIER", "")).strip().lower()
                if model_id == "rainbow":
                    pool_agent = RainbowAgent(
                        n_observations,
                        n_actions,
                        config_path=config_path,
                    )
                elif model_id == "ddpg":
                    pool_agent = DDPGAgent(
                        base_obs_space,
                        single_player_action_space,
                        config_path=config_path,
                    )
                elif model_id in {"tdmpc2", "tdmpc"}:
                    pool_agent = TDMPC2Agent(
                        single_player_action_space,
                        base_obs_space,
                        config_path=config_path,
                    )
                else:
                    if self.verbose:
                        print(
                            f"[SelfPlay] unsupported MODEL_IDENTIFIER={model_id} "
                            f"for opponent config {config_path}"
                        )
                    continue

                pool_agent.load_dict(load_path)
                pool_agent = self._strip_agent_training_state(pool_agent)
                pool.append(
                    {
                        "id": f"pool:{fname}",
                        "kind": model_id,
                        "agent": pool_agent,
                        "model_id": model_id,
                        "path": load_path,
                        "config_path": config_path,
                        "filename": fname,
                        "source": source,
                    }
                )
        finally:
            base_env.close()

        return pool

    def _make_basic_opponent_descriptor(self, weak: bool):
        opponent_id = "basic_weak" if weak else "basic_strong"
        return {
            "id": opponent_id,
            "kind": opponent_id,
            "agent": h_env.BasicOpponent(weak=weak),
            "source": "builtin_basic",
            "filename": opponent_id,
        }

    def _clone_opponent_descriptor(self, descriptor):
        if descriptor["kind"] in {"basic_weak", "basic_strong"}:
            return self._make_basic_opponent_descriptor(weak=descriptor["kind"] == "basic_weak")

        cloned = {key: value for key, value in descriptor.items() if key != "agent"}
        try:
            cloned["agent"] = deepcopy(descriptor["agent"])
        except Exception:
            cloned["agent"] = descriptor["agent"]
        return cloned

    def _build_parallel_opponent_schedule(
            self,
            num_envs: int,
            population_path: str):
        if num_envs <= 0:
            return []

        active_pool_path, pool_source = self._active_parallel_pool_path(population_path)
        pool = self._load_population_pool(active_pool_path, source=pool_source)
        base_cycle = [
            self._make_basic_opponent_descriptor(weak=True),
            self._make_basic_opponent_descriptor(weak=False),
        ]
        base_cycle.extend(pool)

        schedule = []
        for env_idx in range(num_envs):
            descriptor = base_cycle[env_idx % len(base_cycle)]
            schedule.append(self._clone_opponent_descriptor(descriptor))
        return schedule

    def _format_parallel_composition(self, schedule):
        if not schedule:
            return "composition=none"

        id_counts = Counter(descriptor["id"] for descriptor in schedule)
        kind_counts = Counter(descriptor["kind"] for descriptor in schedule)
        id_part = ", ".join(f"{opponent_id}={id_counts[opponent_id]}" for opponent_id in sorted(id_counts.keys()))
        kind_part = ", ".join(f"{kind}={kind_counts[kind]}" for kind in sorted(kind_counts.keys()))
        return f"composition_by_id: {id_part} | composition_by_kind: {kind_part}"

    def _format_per_opponent_winrates(self, opponent_stats):
        if not opponent_stats:
            return "per_opp_winrate=none"

        parts = []
        for opponent_id in sorted(opponent_stats.keys()):
            stats = opponent_stats[opponent_id]
            decisive_games = int(stats["wins"]) + int(stats["losses"])
            winrate = (int(stats["wins"]) / decisive_games) if decisive_games > 0 else 0.0
            parts.append(f"{opponent_id}={winrate:.3f}({int(stats['games'])})")
        return "per_opp_winrate: " + ", ".join(parts)

    def _log_parallel_round_summary(self, i_training_round, opponent_stats):
        if not opponent_stats:
            return

        print(f"[SelfPlay][Round {i_training_round + 1}] per-opponent summary")
        for opponent_id in sorted(opponent_stats.keys()):
            stats = opponent_stats[opponent_id]
            decisive_games = int(stats["wins"]) + int(stats["losses"])
            winrate = (int(stats["wins"]) / decisive_games) if decisive_games > 0 else 0.0
            print(
                f"  - {opponent_id} | kind={stats['kind']} | games={int(stats['games'])} "
                f"| W/L/D={int(stats['wins'])}/{int(stats['losses'])}/{int(stats['draws'])} "
                f"| winrate={winrate:.3f}"
            )
    
    def _copy_agent_without_replay_buffer(self):
        detached_attrs = {}
        for attr in ("replay_buffer", "buffer", "action_noise"):
            if hasattr(self.agent, attr):
                detached_attrs[attr] = getattr(self.agent, attr)
                setattr(self.agent, attr, None)
        try:
            copied_agent = deepcopy(self.agent)
        finally:
            for attr, value in detached_attrs.items():
                setattr(self.agent, attr, value)

        for attr in ("replay_buffer", "buffer", "action_noise"):
            if hasattr(copied_agent, attr):
                setattr(copied_agent, attr, None)
        return copied_agent
    
    def _run_self_play_parallel_round(
            self,
            schedule,
            discrete_actions,
            i_training_round,
            start):
        envs = [Envwrapper(h_env.HockeyEnv(), player2=descriptor["agent"], discrete_actions=discrete_actions) for descriptor in schedule]
        num_envs = len(envs)
        target_episodes = self.agent.NUM_EPISODES * num_envs
        ep_rew_per_env = np.zeros(num_envs, dtype=np.float32)
        state = np.asarray([env.reset()[0] for env in envs])
        episodes_finished = 0
        next_basic_eval_episode = 0
        last_logged_episode = -1
        env_opponent_ids = [descriptor["id"] for descriptor in schedule]
        env_opponent_kinds = [descriptor["kind"] for descriptor in schedule]
        opponent_stats = {}
        for opponent_id, opponent_kind in zip(env_opponent_ids, env_opponent_kinds):
            if opponent_id not in opponent_stats:
                opponent_stats[opponent_id] = {
                    "kind": opponent_kind,
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                }

        try:
            while episodes_finished < target_episodes:
                self.agent.cur_episode = episodes_finished
                action = np.asarray(
                    [
                        self.agent.act(envs[env_idx], state[env_idx], episodes_finished, self.statistics)
                        for env_idx in range(num_envs)
                    ]
                )

                next_state = []
                reward = np.zeros(num_envs, dtype=np.float32)
                done = np.zeros(num_envs, dtype=bool)
                infos = [None for _ in range(num_envs)]

                for env_idx, env in enumerate(envs):
                    obs, rew, terminated, truncated, info = env.step(action[env_idx])
                    next_state.append(obs)
                    reward[env_idx] = float(rew)
                    done[env_idx] = bool(terminated or truncated)
                    infos[env_idx] = info
                    self.agent.observe(
                        state[env_idx],
                        action[env_idx],
                        float(rew),
                        obs,
                        bool(terminated),
                    )

                state = np.asarray(next_state)
                ep_rew_per_env += reward

                if episodes_finished >= self.agent.START_TRAINING:
                    self.agent.update(self.statistics)

                if np.any(done):
                    done_indices = np.where(done)[0]
                    for env_idx in done_indices:
                        self.statistics["ep_rew"].append(float(ep_rew_per_env[env_idx]))
                        ep_rew_per_env[env_idx] = 0.0
                        episodes_finished += 1

                        opponent_id = env_opponent_ids[env_idx]
                        stats = opponent_stats[opponent_id]
                        stats["games"] += 1
                        winner = infos[env_idx].get("winner") if infos[env_idx] is not None else None
                        if winner == 1:
                            stats["wins"] += 1
                        elif winner == -1:
                            stats["losses"] += 1
                        else:
                            stats["draws"] += 1

                        if len(self.statistics["ep_rew"]) > self.mavg_window_size + 1:
                            mv_avg_reward = np.mean(self.statistics["ep_rew"][-self.mavg_window_size:-1])
                            self.statistics["mv_avg_rew"].append(mv_avg_reward)

                        reset_state, _ = envs[env_idx].reset()
                        state[env_idx] = reset_state

                if not np.any(done):
                    continue

                while episodes_finished >= next_basic_eval_episode:
                    eval_episode = next_basic_eval_episode
                    agent_basic_eval = self.agent_against_basicopp_eval(self.agent)
                    agent_strong_eval = self.agent_against_basicopp_eval(self.agent, weak=False)
                    self.agent_against_basic_opp.append(agent_basic_eval)
                    self.agent_against_strong_opp.append(agent_strong_eval)
                    if self.verbose:
                        print(
                            f"[SelfPlay][Round {i_training_round + 1}] "
                            f"weak_basicopp_eval@episode={eval_episode} | winrate={agent_basic_eval:.3f} "
                            f"strong_basicopp_eval@episode={eval_episode} | winrate={agent_strong_eval:.3f}"
                        )
                    next_basic_eval_episode += 1000

                if (
                    self.verbose
                    and episodes_finished % 100 == 0
                    and episodes_finished != last_logged_episode
                ):
                    end = time.time()
                    print(
                        f"[SelfPlay][Round {i_training_round + 1}] "
                        f"episode={episodes_finished}/{target_episodes} "
                        f"| parallel_envs={num_envs} | elapsed={end - start:.1f}s "
                        f"| {self._latest_self_play_stats()} | {self._format_per_opponent_winrates(opponent_stats)}"
                    )
                    last_logged_episode = episodes_finished
        finally:
            for env in envs:
                env.env.close()

        return opponent_stats


    def evaluate_agents(
            self,
            population_path):
        
        N = len(os.listdir(population_path))

        results = []
        start = time.time()
        for i in range(N):
            end = time.time()
            print(f"eval agents against opponent {i}, {end-start} seconds passed")
            agent_against_i_opponent = []

            for j in range(i+1, N):
                agent_load_path = os.path.join(population_path, self.MODEL_IDENTIFIER + f"_{j}") + ".pth"
                opponent_load_path = os.path.join(population_path, self.MODEL_IDENTIFIER + f"_{i}") + ".pth"

                self.agent.load_dict(agent_load_path)
                self.opponent.load_dict(opponent_load_path)

                agent_won_proportion = self.agent_against_agent_eval(self.agent, self.opponent)
                
                agent_against_i_opponent.append(agent_won_proportion)

            results.append(agent_against_i_opponent)

        print(results)
        return results


        # TODO: make population path self argument

    def _agent_against_pool_eval(
        self,
        player1, 
        opponents,
        num_episodes = 50):
    
        winrates = []
        for opponent in opponents:
            winrate_against_opponent = self.agent_against_agent_eval(player1, opponent)
            winrates.append(winrate_against_opponent)

        
        winrates.append(self.agent_against_basicopp_eval(self.agent, weak=True, num_episodes=num_episodes))
        winrates.append(self.agent_against_basicopp_eval(self.agent, weak=False, num_episodes=num_episodes))

        return np.asarray(winrates)

    def agent_against_agent_eval(
            self, 
            player1, 
            player2= "basicopp",
            environment = 'Hockey-One-v0',
            num_episodes = 50):
        
        if environment == 'Hockey-One-v0':
            env = h_env.HockeyEnv()
            obs, info = env.reset()
            obs_agent2 = env.obs_agent_two()
            score = {"player1": 0, "player2": 0}

            for _ in range(num_episodes):
                d = False
                obs, info = env.reset()
                while not d:
                    a1 = self._resolve_eval_action(player1, env, obs)
                    a2 = self._resolve_eval_action(player2, env, obs_agent2)

                    obs, r, d, _, info = env.step(np.hstack([a1,a2]))   
                    obs_agent2 = env.obs_agent_two()
                if info["winner"] == 1:
                    score["player1"] += 1
                else:
                    score["player2"] += 1

        env.close()

        return score["player1"]/num_episodes
    
    def agent_against_basicopp_eval(
            self, 
            player1,
            weak=True,
            environment = 'Hockey-One-v0',
            num_episodes = 50):
        
        if environment == 'Hockey-One-v0':
            env = h_env.HockeyEnv()
            obs, info = env.reset()
            obs_agent2 = env.obs_agent_two()
            score = {"player1": 0, "player2": 0}

            
            player2 = h_env.BasicOpponent(weak=weak)

            for _ in range(num_episodes):
                d = False
                obs, info = env.reset()
                while not d:
                    a1 = self._resolve_eval_action(player1, env, obs)

                    a2 = player2.act(obs_agent2)

                    obs, r, d, _, info = env.step(np.hstack([a1,a2]))   
                    obs_agent2 = env.obs_agent_two()
                if info["winner"] == 1:
                    score["player1"] += 1
                else:
                    score["player2"] += 1

        env.close()

        return score["player1"]/num_episodes
    
    def _resolve_eval_action(self, player, env, obs):
        if isinstance(player, RainbowAgent):
            discrete_action = player.act(env=env, state=obs, greedy=True)
            return self._discrete_to_continuous(discrete_action)
        if isinstance(player, Agent):
            return player.act(env=env, state=obs, greedy=True)
        return player.act(obs)
    
    def _latest_self_play_stats(self):
        episodes_seen = max(0, len(self.statistics["ep_rew"]) - 1)
        parts = [f"episodes_seen={episodes_seen}", f"opponents_in_pool={self.population_size}"]

        if len(self.statistics["mv_avg_rew"]) > 0:
            parts.append(f"mv_avg_reward={self.statistics['mv_avg_rew'][-1]:.3f}")
        if len(self.agent_against_basic_opp) > 0:
            parts.append(f"weak_basicopp_winrate={self.agent_against_basic_opp[-1]:.3f}")
        if len(self.agent_against_strong_opp) > 0:
            parts.append(f"strong_basicopp_winrate={self.agent_against_strong_opp[-1]:.3f}")
        if len(self.statistics["tr_loss"]) > 0:
            parts.append(f"last_loss={self.statistics['tr_loss'][-1]:.4f}")
        return " | ".join(parts)
    
    def save_performance_against_basic_opp(self):
        # ------ create performance_against_basic_opp plot ------
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(np.array(self.agent_against_basic_opp ), label="Mean Q", color="blue", linewidth=1.5)
        plt.xlabel("Episodes (1k intervall)")
        plt.ylabel("Proportion of games won")
        plt.title("Performance against basic opponent")
        plt.savefig(os.path.join(self.experiment_path, f"basic_opponent-{self.agent.MODEL_IDENTIFIER}.png"), dpi=300)
        plt.close()
        # ------ create performance_against_strong_opp plot ------
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(np.array(self.agent_against_strong_opp ), label="Mean Q", color="blue", linewidth=1.5)
        plt.xlabel("Episodes (1k intervall)")
        plt.ylabel("Proportion of games won")
        plt.title("Performance against strong opponent")
        plt.savefig(os.path.join(self.experiment_path, f"strong_opponent-{self.agent.MODEL_IDENTIFIER}.png"), dpi=300)
        plt.close()
    
    def _discrete_to_continuous(self, discrete_action):
        """ discrete actions from 0 to 7
        Args:
            action: The discrete action
        Returns:
            continuous action
        """
        action_cont = [(discrete_action == 1) * -1.0 + (discrete_action == 2) * 1.0,  # player x
                   (discrete_action == 3) * -1.0 + (discrete_action == 4) * 1.0,  # player y
                   (discrete_action == 5) * -1.0 + (discrete_action == 6) * 1.0]  # player angle
        
        action_cont.append((discrete_action == 7) * 1.0)

        return action_cont

    def save_to_population(
            self,
            population_path,
            i_training_round):

        os.makedirs(population_path, exist_ok=True)
        save_index = self.population_size
        self.agent.save_dict(population_path, identifier_extension=f"_{save_index}")
        self.population_size += 1
        if self.verbose:
            round_label = "initial_seed" if (save_index == 0 and i_training_round == 0) else str(i_training_round + 1)
            print(
                f"[SelfPlay] saved_opponent_snapshot={save_index} "
                f"| from_round={round_label} | population_size={self.population_size}"
            )
    
    def select_from_population(
            self,
            population_path):
        
        if not self.fixed_opponents:
            p = random.random()
            pool = self._load_population_pool(population_path, source="self_play_pool")

            if p < 0.2 or not pool:
                self.opponent = h_env.BasicOpponent(weak=False)
                self.last_selected_opponent_info = "strong_basic_opponent"
                return

            if p < 0.5:
                recent_pool = pool[-3:] if len(pool) >= 3 else pool
                chosen = random.choice(recent_pool)
                self.opponent = chosen["agent"]
                self.last_selected_opponent_info = f"recent_pool_{chosen['model_id']}_{chosen['filename']}"
                return

            chosen = random.choice(pool)
            self.opponent = chosen["agent"]
            self.last_selected_opponent_info = f"random_pool_{chosen['model_id']}_{chosen['filename']}"
        
        else:
            p = random.random()
            if p < 0.2:
                self.opponent = h_env.BasicOpponent(weak=False)
                self.last_selected_opponent_info = "fixed_pool_strong_basic"
            elif p < 0.4:
                self.opponent = h_env.BasicOpponent(weak=True)
                self.last_selected_opponent_info = "fixed_pool_weak_basic"
            else:
                pool = self._load_population_pool(self.fixed_opponents_path, source="fixed_pool")
                if pool:
                    chosen = random.choice(pool)
                    self.opponent = chosen["agent"]
                    self.last_selected_opponent_info = f"fixed_pool_{chosen['model_id']}_{chosen['filename']}"
                else:
                    p = random.random()
                    if p < 0.5:
                        self.opponent = h_env.BasicOpponent(weak=False)
                        self.last_selected_opponent_info = "fallback_strong_basic"
                    else:
                        self.opponent = h_env.BasicOpponent(weak=True)
                        self.last_selected_opponent_info = "fallback_weak_basic"


    def save_performance_against_fixed_pool(self):
            self.all_winrates_vs_pool = np.vstack(self.all_winrates_vs_pool)


            N = self.all_winrates_vs_pool.shape[1]
            colors = plt.cm.viridis(np.linspace(0, 1, N))

            for i in range(N):
                all_winrates_vs_i = self.all_winrates_vs_pool[:, i]
                plt.plot(all_winrates_vs_i, label=f"{self.fixed_opponents_labels[i]}", color=colors[i], linewidth=1.5)
            
            plt.xlabel("Evaluation intervalls")
            plt.ylabel("Average Agent Winrate vs Opponents")
            plt.title("Performance against Fixed Opponent pool")
            plt.legend()
            plt.savefig(os.path.join(self.experiment_path, f"opponent_pool_performance-{self.agent.MODEL_IDENTIFIER}.png"), dpi=300)
            plt.close()




    def save(self, *args, **kwargs):
        pass
