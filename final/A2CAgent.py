import os.path
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import gym
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from datetime import datetime
import json
from tqdm import trange
import itertools
import operator
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# todo forgot to add the rewards and opponent observation


def accumulate(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
        yield key, np.mean([item[1] for item in subiter])



logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_actor = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(logdir, 'actor'))
tensorboard_callback_critic = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(logdir, 'critic'))

# HYPERPARAMETERS
# VAE_fc1_dims = 10
# VAE_LSTM_dims = 20
# VAE_latent_dims = 2
# A2C_fc1_dims = 20
# A2C_fc2_dims = 5
# alpha = 0.0002
# beta = 0.0010
# from kerastuner import HyperModel
# class SMA2CHyperModel(HyperModel):
#     pass




def actor_critic(obs_shape, action_space, config):
    """
    A2C without the encoder
    """
    a2c_fc = config.get('a2c_fc', [20, 5])
    alpha = config.get('alpha', 0.0002)
    beta = config.get('beta', 0.0010)
    assert len(a2c_fc) > 0, f"can't have empty length"
    obs_inputs = Input(shape=obs_shape, batch_size=1, name='observation_input')
    act_hidden = Dense(a2c_fc[0], name='actor_hidden0')(obs_inputs)
    critic_hidden = Dense(a2c_fc[0], name='critic_hidden0')(obs_inputs)
    for i, layer_dim in enumerate(a2c_fc[1:]):
        act_hidden = Dense(layer_dim, name=f'actor_hidden{i+1}')(act_hidden)
        critic_hidden = Dense(layer_dim, name=f'critic_hidden{i+1}')(critic_hidden)
    probs = Dense(action_space, activation='softmax',
                  name='action_probabilities')(act_hidden)
    values = Dense(1, activation='linear', name='value')(critic_hidden)
    actor = Model(inputs=obs_inputs, outputs=probs)
    critic = Model(inputs=obs_inputs, outputs=values)

    actor.compile(optimizer=Adam(learning_rate=alpha),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    critic.compile(optimizer=Adam(learning_rate=beta), loss='mse')
    actor.summary()
    critic.summary()
    return actor, critic



class A2CAgent():

    def __init__(self, env, config):
        # Initialization
        
        print(f'config was as follows: {config}')
        self.env: gym.Env = env
        self.action_size = self.env.action_space.n
        self.histories = []
        self.memory = []

        self.save_path = config.get("output", 'models')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.path = f'{self.env.name}_SMA2C'
        self.model_name = os.path.join(self.save_path, self.path)

        self.actor, self.critic = actor_critic(self.env.observation_space.n, self.env.action_space.n, config)

        # self.actor, self.critic, self.encoder = \
        #     sma2c(self.env.observation_space.n + 2 + 1,
        #           self.env.action_space.n)

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

        # print(f'Input to encoder: {ins}, observation: {observation}')
        probs = self.actor.predict(ins[np.newaxis, :self.env.observation_space.n])[0]
        # action = np.random.choice(self.action_size)
        action = np.random.choice(self.action_size, p=probs) if not np.isnan(
            probs).any() else np.random.choice(self.action_size)
        # action = np.argmax(probs)
        return action

    def step(self, action):
        return self.env.step(action)

    def remember(self, obs, action, reward):
        self.memory.append((obs, action, reward))

    def load(self):
        self.actor.load_weights('actor.h5')
        self.critic.load_weights('critic.h5')
        self.encoder.load_weights('encoder.h5')

    def save(self):
        # del kwargs
        self.actor.save(os.path.join(self.save_path, 'actor.h5'))
        self.critic.save(os.path.join(self.save_path, 'critic.h5'))
        # self.encoder.save(os.path.join(self.save_path, 'encoder.h5'))
        #json.dump(self.memory, open('memory.json', 'w'))

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

    def replay(self):
        observations, actions, rewards = zip(*self.memory)
        # reshape appropriately
        discounted_rewards = self.discount_rewards(rewards)
        # values = []
        # encoder_ins = []
        # TODO first one
        first_input = np.zeros(8)
        first_input[observations[0]] = 1.0
        a2c_inputs = [(first_input[:], first_input[:5])]
        for i in range(1, len(self.memory)):
            inputs = np.zeros(8)
            inputs[observations[i]] = 1.0
            inputs[-3 + actions[i-1]] = 1.0
            inputs[-1] = rewards[i-1]
            # encoder_ins.append(inputs)
            a2c_inputs.append((inputs[:], inputs[:5]))

        enc, obs = zip(*a2c_inputs)
        enc, obs = np.array(enc), np.array(obs)
        # print(f'shapes: {enc.shape}, {obs.shape}')
        # open('enc.txt', 'w').write(enc.__str__())
        values = self.critic.predict(obs, batch_size=1)[:, 0]
        # values = np.array(values)
        # print(f'values from the critic: {values}')
        advantages = discounted_rewards - values
        # print(f'advantages: {advantages}')
        # convert actions to OHE
        acts = []
        for action in actions:
            if action:
                acts.append([0., 1.])
            else:
                acts.append([1., 0.])
        acts = np.array(acts)
        # fit the data
        his = self.actor.fit(
            obs, acts, sample_weight=advantages, batch_size=1, epochs=1, verbose=0)
        self.critic.fit(obs, discounted_rewards,
                        batch_size=1, epochs=1, verbose=0)
        self.histories.append(his)
        # clear the memory
        self.memory.clear()

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
            self.replay()
            t.set_description(
                f"Running agent ({real_score} vs {self.env.opponent.name})")
            t.refresh()  # to show immediately the update
        print()
        print(f'running complete')
        json.dump(scores, open(os.path.join(
            self.save_path, 'scores.json'), 'w'))
        json.dump(real_scores, open(os.path.join(
            self.save_path, 'real_scores.json'), 'w'))
        print(list(accumulate(sorted(scores, key=lambda x: x[0]))))
        
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


