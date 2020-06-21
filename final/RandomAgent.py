import json
from tqdm import trange
import numpy as np
import os
class RandomAgent():

    def __init__(self, env, config):
        # Initialization
        
        print(f'config was as follows: {config}')
        self.env: gym.Env = env
        self.action_size = self.env.action_space.n
        self.histories = []
        self.memory = []
        self.save_path = config.get("output", 'models')


    def reset(self, opponent=None):
        observation = self.env.reset(opponent)
        # self.encoder.reset_states()  # TODO should we do this here?
        return observation

    
    def act(self, obs, act_prev, reward_prev, done_prev):
        # if done_prev: # reset states if we are done with a batch
        #     self.encoder.reset_states()
        #     self.actor.reset_states()
        #     self.critic.reset_states()
        # create input
        ins = np.zeros(self.env.observation_space.n +
                       self.env.action_space.n + 1)
        # set observation
        ins[obs] = 1.0
        # set prev reward
        # TODO normalize?
        ins[-1] = reward_prev
        # set prev action
        ins[-3 + act_prev] = 1.0
        # add axis to be batchlike
        # ins = ins[np.newaxis, :]
        action = np.random.choice(self.action_size)

        return action

    def step(self, action):
        return self.env.step(action)

    def remember(self, obs, action, reward):
        self.memory.append((obs, action, reward))


    def discount_rewards(self, rewards):
        gamma = 0.99
        running_add = 0
        discounted_rewards = np.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            running_add = (running_add * gamma) + rewards[i]
            discounted_rewards[i] = running_add

        # normalize
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards


    def save(self, *args, **kwargs):
        pass
    def run(self, episodes=100):
        scores = []
        real_scores = []
        t = trange(episodes, desc='Running agent', leave=True)
        # t = range(episodes)
        for e in t:
            obs = self.reset()
            done, score, real_score = False, 0, 0
            prev_action, prev_reward, prev_done = 0, 0, True

            while not done:
                action = self.act(obs, prev_action, prev_reward, prev_done)
                # step
                next_obs, reward, done, _ = self.step(action)
                # print(f'obtained {reward} by playing {action}')
                # save this for batch learning
                self.remember(obs, action, reward)
                prev_action = action
                prev_done = False
                prev_reward = reward
                obs = next_obs
                score += reward
                real_reward = self.env.scaler.inverse_transform([[reward]])[0][0]
                # assert real_reward in [0,-1,2,3], f'how can real reward be {real_reward}'
                real_score += real_reward
            

            real_scores.append((self.env.opponent.name, real_score))
            scores.append((self.env.opponent.name, score))
            # print(f'episode: {e}/{episodes}, {score}, {self.env.opponent.name}')
            # print(self.memory)
            
            t.set_description(
                f"Running agent ({real_score} vs {self.env.opponent.name})")
            t.refresh()  # to show immediately the update
        print()
        print(f'running complete')
        json.dump(scores, open(os.path.join(
            self.save_path, 'scores.json'), 'w'))
        json.dump(real_scores, open(os.path.join(
            self.save_path, 'real_scores.json'), 'w'))
        
    def test(self, episodes=10):
        scores = []
        for e in range(episodes):
            state = self.reset()
            done = False
            score = 0
            while not done:
                action = np.argmax(self.actor.predict(state))
                state, reward, done, _ = self.step(action)
                score += reward

            print("episode: {}/{}, score: {}".format(e, episodes, score))
            scores.append(score)
        print(f'testing complete, AVG SCORE: {np.mean(scores)}')
