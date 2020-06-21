from SMA2CAgent import SMA2CAgent
from A2CAgent import A2CAgent
from RandomAgent import RandomAgent
# from .SMA2CAgent import SMA2CAgent
import gym
import numpy as np
from IPD_fixed import IPDEnv
import axelrod
import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", help='number of rounds to play per episode', type=int, default=20)
    parser.add_argument("--episodes", help='number of episodes to play', type=int, default=1000)
    parser.add_argument("--seed", help='random seed, -1 if random', type=int, default=-1)
    parser.add_argument("--output", help="output folder", default=f'output-{time.time():.0f}')
    parser.add_argument("--pure-a2c", help="Don't use an encoder", action='store_true')
    parser.add_argument("--alpha", help='LR of encoder', type=float)
    parser.add_argument("--beta", help = 'LR of A2C agent', type=float)
    parser.add_argument("--lstm-dims", help='LSTM dimensions', type=int)
    parser.add_argument("--encoder-fc", help='dimensions of encoder dense layers',type=int, action='append')
    parser.add_argument("--a2c-fc", help='dimensions of a2c hidden layers', type=int, action='append')
    parser.add_argument("--latent-dims", help='dimensions of code', type=int)
    parser.add_argument("opponents", help='opponents that the bot should face', nargs="*")
    parser.add_argument("--random", help="Don't use an agent, just random", action='store_true')
   
    # parser.add_argument("")
    args = parser.parse_args()
    opponents = []
    strats = dict([(s.name.lower(), s) for s in axelrod.all_strategies])
    for opp in args.opponents:
        if opp not in strats:
            print(f'{opp} not found in strats')
        s = strats[opp]
        opponents.append(s)

    env = IPDEnv({'rounds': args.rounds, 'opponents' : opponents})
    seed = args.seed if args.seed != -1 else None
    env.seed(seed=seed)
    # remove empty values
    config = {k: v for k, v in vars(args).items() if v is not None}
    if config['pure_a2c']:
        print("____USING PURE A2C_____")
        agent= A2CAgent(env, config)
    elif config['random']:
        print("__RANDOM AGENT___")
        agent = RandomAgent(env, config)
    else:
        print("____USING SMA2C______")
        agent = SMA2CAgent(env, config)
    # obs = env.reset()
    # action = agent.act(obs, 0, 0, 1)
    # print(f'resulting action: {action}')
    # encodings_before = np.array(agent.encode_run(axelrod.Cooperator()))
    # print(f'encodings before: {encodings_before}')
    agent.run(episodes=args.episodes)
    # encodings_after_c = np.array(agent.encode_run(axelrod.Cooperator()))
    # encodings_after_d = np.array(agent.encode_run(axelrod.Defector()))
    # print(f'encodings after: {encodings_after_c}')
    # print(encodings_after_d)
     
    agent.save()
