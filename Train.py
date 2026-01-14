import numpy as np
import torch
from agents.Agents import Agent


class Training():
    def __init__(
        self,
        agent = ,
        env,
        verbose=False
        ):
        
        self.statistics: np.ndarray = None
        self.agent: Agent = None
        self.env = env
        self.verbose = verbose

    
    def train(self):
        # set float as default
        torch.set_default_dtype(torch.float32)
        
        if torch.cuda.is_available():
            print("\nUsing CUDA.")
            print (torch.version.cuda,"\n")
        else:
            print ("\nNot using CUDA.\n")

        # print hyperparameter settings to console via Agent.printHyper()??

        # for hyperparameter tuning: save model to subfolder where the subfolder name specifies hyperparameter used

        optimizer = optim.Adam(policy_net.parameters(), self.agent.Hyperparameters["lr"])

        obs, info = env.reset()
        

    
    def plot(self, *args, **kwargs):
        pass

    
    def train_self_play(self, *args, **kwargs):
        pass

    
    def save(self, *args, **kwargs):
        pass
