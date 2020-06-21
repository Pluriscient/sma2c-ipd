import gym
import axelrod
from axelrod.player import Player
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
C, D = axelrod.Action.C, axelrod.Action.D
# be ready for *any* strategies
# opponents = [c() for c in axelrod.all_strategies]
# opponents = [axelrod.MetaMinority()]
DEFAULT_OPPONENTS = [
    axelrod.Cooperator,
    # axelrod.Defector,
    # axelrod.Detective,
    axelrod.TitForTat
]


class IPDPlayer(Player):
    name = "hopeful"
    classifier = {
        "memory_depth": 1,  # Four-Vector = (1.,0.,1.,0.)
        "stochastic": False,
        "makes_use_of": set(),
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self):
        super().__init__()


class IPDEnv(gym.Env):
    name = "IPD"
    metadata = {'render.modes': []}
    reward_range = (float(0), float(3))
    action_space = Discrete(2)


    def __init__(self, env_config):
        super().__init__()
        opps = env_config.get("opponents") or DEFAULT_OPPONENTS
        self.opponents = [c() for c in opps]
        assert all(isinstance(o, Player) for o in self.opponents)
        self.own_player = IPDPlayer()
        RSTP = env_config.get("RSTP", [2, -1, 3, 0])
        RSTP = [float(x) for x in RSTP]
        RSTP = np.array(RSTP)
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        
        self.scaler.fit(RSTP.reshape((-1,1)))

        RSTP = self.scaler.transform(RSTP.reshape(-1,1)).reshape(-1)

        # scaler.inverse_transform(res)
        # RSTP = scale(RSTP)
        # print(f'rstp: {RSTP}')
        self.rng = np.random.default_rng()
        
        self.payout_mat = np.array([[RSTP[0], RSTP[2]], [RSTP[1], RSTP[3]]])
        
        # self.payout_mat = scale(self.payout_mat, with_std=False)
        # print(f'payout matrix: {self.payout_mat}')
        self.rounds_per_episode = env_config.get("rounds", 100)
        self.observation_space = Discrete(5)
        # self.observation_space = Disc(5)

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        

    def reset(self, opponent=None):
        self.step_count = 0
        self.opponent = self.rng.choice(self.opponents) if opponent is None else opponent
        init_joint_action = 4
        observation = init_joint_action
        return observation

    def step(self, ac0):
        """
        action: 0 if cooperating, 1 if defecting
        """
        opponent_action = self.opponent.strategy(self.own_player)
        own_action = C if ac0 == 0 else D
        # print(f'actions: {own_action} vs {opponent_action}')
        self.opponent.update_history(opponent_action, own_action)
        self.own_player.history.append(own_action, opponent_action)
        ac1 = 0 if opponent_action == C else 1
        rewards = [self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]]
        joint_action = ac0 * 2 + ac1
        self.step_count += 1
        done = self.step_count == self.rounds_per_episode
        return joint_action, rewards[0], done, {}
