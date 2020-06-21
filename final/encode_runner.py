import axelrod as axl
import numpy as np
from SMA2CAgent import SMA2CAgent
# from .SMA2CAgent import SMA2CAgent
import gym
import numpy as np
from IPD_fixed import IPDEnv
import axelrod
import time
import argparse
import json
import pandas as pd
from pandas import DataFrame as df

env = IPDEnv({})
# remove empty values
config = {}
agent= SMA2CAgent(env, config)



encodings_before_c = df(np.array(agent.encode_run(axelrod.Cooperator())))
encodings_before_d = df(np.array(agent.encode_run(axl.Defector())))
encodings_before_tft = df(np.array(agent.encode_run(axl.TitForTat())))
print(encodings_before_c.shape)
agent.load('outputs/1592391802-sma2c-4000-smaller-mid')
encodings_after_c = df(np.array(agent.encode_run(axelrod.Cooperator())))
encodings_after_d = df(np.array(agent.encode_run(axelrod.Defector())))
encodings_after_tft = df(np.array(agent.encode_run(axl.TitForTat())))
encodings = [encodings_before_c, encodings_before_d, encodings_before_tft, encodings_after_c, encodings_after_d, encodings_after_tft]
for i, df in enumerate(encodings):
    df.to_csv(f'{i}.csv')


# json.dump([x.tolist() for x in [encodings_before, encodings_after_c, encodings_after_d]], open('encodings.json', 'w'))
