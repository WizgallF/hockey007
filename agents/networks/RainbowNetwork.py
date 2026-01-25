import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class RainbowNetwork(nn.Module):
    def __init__(self, observation_size, action_size, device, dueling = True, noisy=True, distributional_q = True, n_atoms=201, sigma0=0.5):
        """ Create RainbowNetwork
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.observation_size = observation_size
        self.action_size = action_size
        self.noisy = noisy
        self.dueling = dueling
        self.distributional_q = distributional_q
        self.sigma0 = sigma0

        self.n_atoms = n_atoms if self.distributional_q else 1

        
        # MLP
        hidden_1_dim = 128
        hidden_2_dim = 128

        
        #  noisy net linear layers 
        self.features =  nn.Sequential(NoisyLinearLayer(observation_size, hidden_1_dim, self.noisy, self.sigma0), nn.ReLU())

        if self.dueling:
            self.value = nn.Sequential(
                NoisyLinearLayer(hidden_1_dim, hidden_2_dim, self.noisy, self.sigma0), nn.ReLU(),
                NoisyLinearLayer(hidden_2_dim, self.n_atoms, self.noisy, self.sigma0)
            )

        self.advantage = nn.Sequential(
            NoisyLinearLayer(hidden_1_dim, hidden_2_dim, self.noisy, self.sigma0), nn.ReLU(),
            NoisyLinearLayer(hidden_2_dim, action_size * self.n_atoms, self.noisy, self.sigma0)
        )
            


        

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """
        observation = torch.Tensor(observation).to(self.device)
        batch_size = observation.shape[0]

        
        features = self.features(observation)

        if self.dueling:
            V = self.value(features)
            A = self.advantage(features)
            
            if self.n_atoms != 1:
                A = A.view(batch_size, self.action_size, self.n_atoms)
                # Broadcast V to [B, A, N]
                V = V.unsqueeze(1)                               # [B, 1, N]

            logits = V + (A - A.mean(dim=1, keepdim=True))   # [B, A, N]
        else:
            A = self.advantage(features)
            if self.n_atoms != 1:
                A = A.view(batch_size, self.action_size, self.n_atoms)
            logits = A



        return logits
    

class NoisyLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, noisy = True, sigma0=0.5):
        """Factorised Noisy Linear layer
        Parameters
        ----------
        in_features: input dim
        out_features: output dim
        sigma0: initial sigma scaling
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0
        self.noisy = noisy

        # weight / bias means
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))

        # weight / bias sigmas 
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 2.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.mu_weight, -bound, bound)
        nn.init.uniform_(self.mu_bias, -bound, bound)
        # initialize sigma to sigma0 / sqrt(in)
        self.sigma_weight.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma0 / math.sqrt(self.out_features))

    def _scaled_noise(self, size):
        x = torch.randn(size, device=self.mu_weight.device)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training and self.noisy:
            eps_in = self._scaled_noise(self.in_features)   # (in,)
            eps_out = self._scaled_noise(self.out_features) # (out,)
            # factorised outer product
            eps_w = torch.outer(eps_out, eps_in)            # (out, in)
            eps_b = eps_out                                # (out,)
            weight = self.mu_weight + self.sigma_weight * eps_w
            bias = self.mu_bias + self.sigma_bias * eps_b
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)