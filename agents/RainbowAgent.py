from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import yaml
import os
import json
import random
import torch.optim as optim
from datetime import datetime
from agents.AgentBaseclass import Agent
from agents.networks.RainbowNetwork import RainbowNetwork
from agents.utils.RainbowUtils import ReplayBuffer, LinearSchedule

torch.set_num_threads(1)

class RainbowAgent(Agent):
    """
    A Rainbow Q-Learning agent implementation based on the Agent Baseclass.
    """
    def __init__(
            self,
            n_observations: int,
            n_actions: int,
            verbose = False,
            config_path: str = "configs/rainbow_config.yaml",
            eval_mode = True
            ):

        # ------ load configs from "rainbow_config.yaml" ------
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.configs = config
        self.__dict__.update(config)
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.verbose = verbose
        self.t = 0

        # ------ initialize neural networks ------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = RainbowNetwork(n_observations, n_actions, self.device, self.DUELING, self.NOISY, self.DISTRIBUTIONAL_Q, n_atoms=self.N_ATOMS, sigma0=self.SIGMA0, hidden_1_dim=self.HIDDEN_1_DIM, hidden_2_dim=self.HIDDEN_2_DIM).to(self.device)
        self.target_net = RainbowNetwork(n_observations, n_actions, self.device, self.DUELING, self.NOISY, self.DISTRIBUTIONAL_Q, n_atoms=self.N_ATOMS, sigma0=self.SIGMA0, hidden_1_dim=self.HIDDEN_1_DIM, hidden_2_dim=self.HIDDEN_2_DIM).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.cur_episode = 0

        if eval_mode:
            self.policy_net.eval()

        # ------ optimizer ------
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR, betas=(self.ADAM_BETA_1, self.ADAM_BETA_2), eps=self.ADAM_EPS)

        # ------ replay buffer ------
        self.replay_buffer = ReplayBuffer(self.BUFFER_SIZE, 
                                   self.GAMMA, 
                                   n_step=self.N_STEP, 
                                   prioritized=self.PRIORITIZED_REPLAY, 
                                   alpha=self.PR_ALPHA, 
                                   beta=self.PR_BETA_START,
                                   prio_clipping = self.PRIO_CLIP,
                                   prio_clip_value = self.PRIO_CLIP_VALUE)
        
        # ------ beta scheduler ------
        self.beta_schedule = LinearSchedule(schedule_timesteps=self.NUM_EPISODES * self.BETA_TARGET_REACHED * int(self.BETA_SCHEDULE_ACTIVE), initial_p=self.PR_BETA_START, final_p=1.0)

        # ------ support ------
        if self.DISTRIBUTIONAL_Q:
            self.support, self.delta_z = self._c51_support()
        else:
            self.support = None
            self.delta_z = None

    
    def act(
            self,
            env,
            state: np.ndarray,
            i_episode: int = 0,
            statistics: dict[str, list] = None,
            greedy: bool = False
            ) -> int:
        """
        Given an observation, selects a discrete action. 
        Uses the epsilon-greedy strategy if network weights 
        are not noisy and greedy strategy is not used.
        ------------
        Parameters:
            env: The environment
            state: The current state
            i_episode: Number of episodes already seen
            statistics: The training statistics
            greedy: Whether greedy strategy for action selection is used
        -----------
        Return:
            Integer specifying the selected action.
        """

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * i_episode * 0.69 / (self.EPS_DECAY * self.NUM_EPISODES)) # after EPS_DECAY proportion of episodes, the schedule divides eps/2 as ln(1/2) = -0.69
        
        if sample > eps_threshold or self.NOISY or greedy:

            # ------ greedy action selection ------
            if self.DISTRIBUTIONAL_Q:
                with torch.no_grad():
                    logits = self.policy_net(state)                          
                self.support = self.support.to(logits.device)

                probs  = torch.softmax(logits, dim=-1)
                q_vals = (probs * self.support.view(1, 1, -1)).sum(-1)
                if state.shape[0] == 1:
                    return q_vals.argmax(dim=1).squeeze().item()
                else:
                    q_vals.argmax(dim=1).squeeze().detach().cpu().numpy()
            
            else:
                with torch.no_grad():
                    q_values = self.policy_net(state)
                
                actions = torch.argmax(q_values, dim=1).detach().cpu().numpy()
                if len(actions) == 1:
                    return actions[0]
                else:
                    return actions
        else:

            # ------ random action sampling from environment ------
            if state.shape[0] == 1:
                return env.action_space.sample()
            else:
                random_actions = [env.action_space.sample() for _ in range(state.shape[0])]
                return np.asarray(random_actions)
            

    def observe(
            self,
            state: np.ndarray, 
            action: int, 
            reward: int, 
            next_state: np.ndarray,
            terminated: bool
            ):
        """
        Saves observed transition in the replay buffer.
        ----------
        Parameters:
            state: Current state
            action: The action selected in current state
            reward: The reward given in current state for the selected action
            next_state: The next state of the environment
            terminated: Whether the episode has terminated after current transition

        """
        if state.ndim == 1:
            self.replay_buffer.add(np.expand_dims(state, axis=0), np.asarray(action).squeeze(), np.asarray([reward]), np.expand_dims(next_state, axis=0), int(terminated))
        else:
            for idx in range(state.shape[0]):
                self.replay_buffer.add(np.expand_dims(state[idx], axis=0), np.asarray(action[idx]).squeeze(), np.asarray([reward[idx]]), np.expand_dims(next_state[idx], axis=0), int(terminated[idx]))

    
    def update(
            self,
            statistics: dict[str, list] = None
            ):
        """
        One training iteration of Ranbow Q-Learning with BATCH_SIZE samples drawn from the replay buffer.
        ----------
        Parameters:
            statistics: The training statistics
        """

        if self.PRIORITIZED_REPLAY:
                self.replay_buffer.set_beta(self.beta_schedule.value(self.cur_episode))

        sample = self.replay_buffer.sample(self.BATCH_SIZE)

        # Robust to prio vs non-prio replay buffer
        if len(sample) == 5:
            state_batch, action_batch, reward_batch, next_states, dones = sample
            weights = np.ones_like(reward_batch, dtype=np.float32)
            idxes = None
        else:
            state_batch, action_batch, reward_batch, next_states, dones, weights, idxes = sample

        state_batch   = torch.from_numpy(state_batch).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        action_batch   = torch.from_numpy(action_batch).long().to(self.device)
        reward_batch   = torch.from_numpy(reward_batch).float().to(self.device)
        dones     = torch.from_numpy(dones).float().to(self.device)
        weights_t = torch.from_numpy(weights).float().to(self.device)
        
        
        non_final_mask = torch.tensor(tuple(map(lambda done: not bool(done),
                                            dones)), device=self.device, dtype=torch.bool)
       
        non_final_next_states = torch.vstack([s for i, s in enumerate(next_states) if non_final_mask[i].item()])

        if not self.DISTRIBUTIONAL_Q:
            all_state_action_values = self.policy_net(state_batch)
            state_action_values = all_state_action_values.gather(1, action_batch.unsqueeze(1))

        
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device).unsqueeze(1)
            with torch.no_grad():
                if self.DOUBLEQ_LEARNING:
                    next_state_actions = self.policy_net(non_final_next_states).argmax(1, keepdim=True)
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions)
                else:
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1, keepdim=True).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * (self.GAMMA) ** self.N_STEP) + reward_batch

            # Compute Huber loss
            criterion = torch.nn.SmoothL1Loss(reduction='none')
            per_sample_loss = criterion(state_action_values, expected_state_action_values).squeeze()
            loss = (per_sample_loss * weights_t).mean()

        else:
            
            all_state_action_values = self.policy_net(state_batch)
            log_probs = F.log_softmax(all_state_action_values, dim=-1)                 
            log_probs_sa = log_probs.gather(
                1, action_batch.view(-1, 1, 1).expand(-1, 1, self.N_ATOMS)
            ).squeeze(1)

            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device).unsqueeze(1)

            with torch.no_grad():
                # --- Choose next action  ---
                if self.DOUBLEQ_LEARNING:
                    # use policy network
                    next_logits_online = self.policy_net(next_states) 
                    next_probs_online = F.softmax(next_logits_online, dim=-1)
                    next_q_online = (next_probs_online * self.support.view(1, 1, -1)).sum(dim=-1)  
                    next_actions = next_q_online.argmax(dim=1)
                else:
                    # use target network
                    next_logits_target = self.target_net(next_states) 
                    next_probs_target = F.softmax(next_logits_target, dim=-1)
                    next_q_target = (next_probs_target * self.support.view(1, 1, -1)).sum(dim=-1)  
                    next_actions = next_q_target.argmax(dim=1)
                

                #  Evaluate next distribution
                next_logits_target = self.target_net(next_states)
                next_probs_target = F.softmax(next_logits_target, dim=-1) 
                next_probs = next_probs_target.gather(
                    1, next_actions.view(-1, 1, 1).expand(-1, 1, self.N_ATOMS)
                ).squeeze(1)

                # Build target distribution
                expected_state_action_values = self._c51_projection(next_probs, reward_batch, non_final_mask)
            

            # cross-entropy loss
            per_sample_loss = -(expected_state_action_values * log_probs_sa).sum(dim=1)
            loss = (per_sample_loss * weights_t).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        q_values_min = all_state_action_values.min(dim=1)[0].mean().item()
        q_values_max = all_state_action_values.max(dim=1)[0].mean().item()
        q_values_mean = all_state_action_values.mean().item()
        statistics["mean_q"].append(q_values_mean)
        statistics["min_q"].append(q_values_min)
        statistics["max_q"].append(q_values_max)

        loss = float(loss.detach().cpu().item())
        statistics["tr_loss"].append(loss)

        # Prioritized replay buffer update (As in Rainbow paper by Deepmind)
        if idxes is not None and hasattr(self.replay_buffer, "update_priorities"):
            new_priorities = per_sample_loss.detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(idxes, new_priorities)

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        
    def _c51_support(self):
        """
        If distributional Q-Learning is used, creates the support of the atomized probability distribution.
        -----------
        Return:
            Tuple(
                support: The support for the distribution
                delta_z: The spacing between the atoms
                )
        """
        support = torch.linspace(self.VMIN, self.VMAX, self.N_ATOMS, device=self.device)
        delta_z = (self.VMAX - self.VMIN) / (self.N_ATOMS - 1)
        return support, delta_z


    def _c51_projection(
            self, 
            next_probs: torch.Tensor, 
            rewards: torch.Tensor, 
            non_final_mask: torch.Tensor
            ) -> torch.Tensor:
        """
        If distributional Q-Learning is used, projects the probabilities of the target support 
        (which shrink by gamma and shift by the reward) back on the original support.
        Implementation is based on the C51 paper: https://arxiv.org/pdf/1707.06887 and the library: 
        "PFRL, a deep reinforcement learning library".
        -----------
        Parameters:
            next_probs: The probability distributions of the next states
            rewards: The rewards of the sampled transitions
            non_final_mask: The mask for selecting the transitions that are not terminated
        -----------
        Return:
            The projected target probability distribution.
        """

        B, N = next_probs.shape

        
        gamma = self.GAMMA ** self.N_STEP

        mask = non_final_mask.unsqueeze(1)
        target_support = gamma * self.support.unsqueeze(0)
        
        tz = rewards + mask * target_support

        #tz = rewards + ones.unsqueeze(1) * gamma * self.support.unsqueeze(0)
        tz = tz.clamp(self.VMIN, self.VMAX)
        b = (tz - self.VMIN) / self.delta_z
        l = b.floor().long()
        u = (l + 1)
        l = l.clamp(0, N - 1)
        u = u.clamp(0, N - 1)
        m = torch.zeros_like(next_probs)

        offset = (torch.arange(B, device=next_probs.device) * N).unsqueeze(1)
        l_idx = (l + offset).view(-1)
        u_idx = (u + offset).view(-1)
        p = next_probs.view(-1)
        b_flat = b.view(-1)
        l_flat = l.view(-1).float()
        m_flat = m.view(-1)
        m_flat.scatter_add_(0, l_idx, p * (l_flat + 1.0 - b_flat))
        m_flat.scatter_add_(0, u_idx, p * (b_flat - l_flat))
        return m


    def save_dict(
            self,
            save_path: str = "",
            identifier_extension: str = ""
            ):
        """
        Save the models state dict to specified path.
        ----------
        Parameter:
            save_path: The path where the model's state dictionary will be saved
            identifier_extension: Add additional information to the filename
        """

        saving_dir = os.path.join(save_path, self.MODEL_IDENTIFIER + identifier_extension + ".pth")
        torch.save(self.policy_net.state_dict(), saving_dir)
    
    
    def load_dict(
            self,
            load_path: str = ""
            ):
        """
        Load the models state dict from specified path.
        ----------
        Parameter:
            load_path: The path from which the model's state dictionary will be loaded
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint)


    def save_experiment_config(
            self, 
            base_dir: str = ""
            ):
        """
        Creates a unique folder name and saves experiment configs to it.
        ----------
        Parameter:
            base_dir: The base directory for the experiment folder
        """
        # Create a unique folder name using timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{timestamp}_{self.MODEL_IDENTIFIER}"
        experiment_path = os.path.join(base_dir, folder_name)
        
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save configs to experiment folder
        with open(f"{experiment_path}/config.yaml", 'w') as f:
            yaml.dump(self.configs, f)
            
        return experiment_path
    

    def print_config(self):
        """
        Prints the config to the terminal.
        """
        # indent=4 makes it look like a structured config file
        pretty_conf = json.dumps(self.configs, indent=4)
        print(f"Loading Agent: {self.MODEL_IDENTIFIER}")
        print(pretty_conf)
        
    

