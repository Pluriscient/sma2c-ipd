from axelrod import Player
import axelrod
from SMA2CAgent import SMA2CAgent
from IPD_fixed import IPDEnv
import os
import numpy as np
C, D = axelrod.Action.C, axelrod.Action.D



class SMA2CPlayer(Player):
    name = "sma2c"
    agent = None
    classifier = {
        "memory_depth": 1,  # Four-Vector = (1.,0.,1.,0.)
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    @staticmethod
    def load_cached(reload, model_folder):
        if reload or SMA2CPlayer.agent is None:
            assert all(elem in os.listdir(
                model_folder) for elem in ['actor.h5', 'critic.h5', 'encoder.h5']), f'models not in {model_folder}, only found {os.listdir(model_folder)}'
            RSTP = [2, -1, 3, 0]
            RSTP = [float(x) for x in RSTP]
            env = IPDEnv({})
            payout_mat = np.array([[RSTP[0], RSTP[2]], [RSTP[1], RSTP[3]]])
            SMA2CPlayer.agent = SMA2CAgent(env, {})
            SMA2CPlayer.agent.load(model_folder)
        return SMA2CPlayer.agent
        
    def __init__(self, model_folder, reload=False):
        super(SMA2CPlayer, self).__init__()
        
        self.agent = SMA2CPlayer.load_cached(reload, model_folder)
        RSTP = [2, -1, 3, 0]
        RSTP = [float(x) for x in RSTP]
        self.payout_mat = np.array([[RSTP[0], RSTP[2]], [RSTP[1], RSTP[3]]])
        self._history = axelrod.History()
        self.name = 'SMA2C'
        
        self.reset()

    def __repr__(self):
        return "SMA2C"

    def reset(self):
        self.done, self.score = True, 0
        self.prev_action, self.prev_reward, self.prev_done = 0, 0, True
        self.obs = self.agent.reset()
        self.history.reset()

    # def receive_match_attributes(self):
    #     # Overwrite this function if your strategy needs
    #     # to make use of match_attributes such as
    #     # the game matrix, the number of rounds or the noise
    #     pass

    def strategy(self, opponent):
        if len(self.history) > 0:
            (own_action, opponent_action) = (self.history[-1], opponent.history[-1])
            ac1 = 0 if opponent_action == C else 1
            ac0 = 0 if own_action == C else 1
            prev_action = ac0
            rewards = [self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]]
            obs = ac0*2 + ac1
            prev_reward = rewards[0]
        else:
            obs = [0,0,0,0,1]
            prev_action = 0
            prev_reward = 0
      
     
        action = self.agent.act(obs, prev_action, prev_reward, self.done)
        self.done = False
        return C if action == 0 else D
