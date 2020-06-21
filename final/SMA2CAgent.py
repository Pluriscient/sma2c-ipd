import os.path
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape, Lambda
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
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

EPS = 1e-5
def accumulate(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
        yield key, np.mean([item[1] for item in subiter])


def sampling(args):
    z_mean, z_log_var = args
    z_log_var = K.clip(z_log_var, 1e-4, 1-1e-4)
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


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


def sma2c_encoder(input_shape, config):
    """
    Forces batch size of 1
    TODO fix that
    """
    encoder_dims = config.get("encoder_fc", [10])
    lstm_dims = config.get("lstm_dims", 20)
    code_dims = config.get("latent_dims", 2)
    inputs = Input(shape=(input_shape,), batch_size=1, name='encoder_input')
    x = inputs
    for i, layer_dims in enumerate(encoder_dims):
        x = Dense(layer_dims, name=f'encoder_fc{i}')(x)
    x = Reshape((encoder_dims[-1], 1), input_shape=(
        encoder_dims[-1],), name='VAE_reshape')(x)
    x = LSTM(lstm_dims, name='VAE_LSTM', stateful=True)(x)
    z_mean = Dense(code_dims, name='z_mean')(x)
    z_log_var = Dense(code_dims, name='z_log_var')(x)
    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    encoder = Model(inputs=inputs, outputs=z)
    return encoder


def sma2c_actor_critic(encoder, obs_shape, action_space, config):
    # get things from the config
    a2c_fc = config.get('a2c_fc', [20, 5])
    alpha = config.get('alpha', 0.0002)
    beta = config.get('beta', 0.0010)
    assert len(a2c_fc) > 0, f"can't have empty length"
    obs_inputs = Input(shape=obs_shape, batch_size=1, name='observation_input')
    x = tf.keras.layers.Concatenate(axis=-1)([encoder.output, obs_inputs])
    act_hidden = Dense(a2c_fc[0], name='actor_hidden0', bias_initializer='random_normal')(x)
    critic_hidden = Dense(a2c_fc[0], name='critic_hidden0')(x)
    for i, layer_dim in enumerate(a2c_fc[1:]):
        act_hidden = Dense(layer_dim, name=f'actor_hidden{i+1}')(act_hidden)
        critic_hidden = Dense(layer_dim, name=f'critic_hidden{i+1}')(critic_hidden)
    probs = Dense(action_space, activation='softmax',
                  name='action_probabilities')(act_hidden)
    values = Dense(1, activation='linear', name='value')(critic_hidden)
    actor = Model(inputs=[encoder.input, obs_inputs], outputs=probs)
    critic = Model(inputs=[encoder.input, obs_inputs], outputs=values)

    actor.compile(optimizer=Adam(learning_rate=alpha),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    critic.compile(optimizer=Adam(learning_rate=beta), loss='mse')
    actor.summary()
    critic.summary()
    return actor, critic


class SMA2CAgent():

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
        self.encoder = sma2c_encoder(
            self.env.observation_space.n + self.env.action_space.n + 1, config)  # reward is last
        self.actor, self.critic = sma2c_actor_critic(
            self.encoder, self.env.observation_space.n, self.env.action_space.n, config)


    def load(self, model_folder):
        self.encoder = load_model(os.path.join(model_folder, 'encoder.h5'))
        self.actor = load_model(os.path.join(model_folder, 'actor.h5'))
        self.critic = load_model(os.path.join(model_folder, 'critic.h5'))

    def reset(self, opponent=None):
        observation = self.env.reset(opponent)
        self.encoder.reset_states()  # TODO should we do this here?
        return observation

    def encode(self, obs, act_prev, reward_prev):
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
        encoding = self.encoder.predict(ins[np.newaxis, :])[0]
        return encoding

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
        probs = self.actor.predict([ins[np.newaxis, :],
                                    ins[np.newaxis, :self.env.observation_space.n]])[0]
        # if np.abs(probs[0] - probs[1]) == 1-1e-1:
        # print(f'difference in probs is very high: ', probs)
        # action = np.random.choice(self.action_size)
        if np.isnan(probs).any():
            print(f'We have {np.count_nonzero(np.isnan(probs))} NANs')
        
        action = np.random.choice(self.action_size, p=probs) if not np.isnan(
            probs).any() else np.random.choice(self.action_size)
        # action = np.argmax(probs)
        # print(f'{probs} -> {action}')
        return action

    def step(self, action):
        return self.env.step(action)

    def remember(self, obs, action, reward):
        self.memory.append((obs, action, reward))



    def save(self):
        # del kwargs
        self.actor.save(os.path.join(self.save_path, 'actor.h5'))
        self.critic.save(os.path.join(self.save_path, 'critic.h5'))
        self.encoder.save(os.path.join(self.save_path, 'encoder.h5'))
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
        discounted_rewards /= np.clip(np.std(discounted_rewards), EPS, 1-EPS)
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
            # print(inputs)
            # batch size has to be 1... but TODO this can be optimized
            # value = self.critic.predict([inputs[np.newaxis, :],
            #                              inputs[np.newaxis, :self.env.observation_space.n]])[:, 0]

        # print(f'encoder ins: {encoder_ins}')
        # print(f'a2c inputs: {a2c_inputs}')

        enc, obs = zip(*a2c_inputs)
        enc, obs = np.array(enc), np.array(obs)
        # print(f'shapes: {enc.shape}, {obs.shape}')
        # open('enc.txt', 'w').write(enc.__str__())
        values = self.critic.predict([enc, obs], batch_size=1)[:, 0]
        # values = np.array(values)
        # print(f'values from the critic: {values}')
        advantages = discounted_rewards - values
        # print(f'advantages: {advantages}')
        # convert actions to OHE
        if np.isnan(advantages).any():
            print('advantages has {} nans', np.count_nonzero(np.isnan(advantages)))
        acts = []
        for action in actions:
            if action:
                acts.append([0., 1.])
            else:
                acts.append([1., 0.])
        acts = np.array(acts)
        # fit the data
        his = self.actor.fit(
            [enc, obs], acts, sample_weight=advantages, batch_size=1, epochs=1, verbose=0)
        self.critic.fit([enc, obs], discounted_rewards,
                        batch_size=1, epochs=1, verbose=0)
        self.histories.append(his)
        # clear the memory
        self.memory.clear()

    def encode_run(self, opponent):
        obs = self.reset(opponent)
        done, score = False, 0
        prev_action, prev_reward, prev_done = 0, 0, True
        encodings = []
        while not done:
            action = self.act(obs, prev_action, prev_reward, prev_done)
            encoding = self.encode(obs, prev_action, prev_reward)
            encodings.append(encoding)
            # step
            next_obs, reward, done, _ = self.step(action)

            prev_action = action
            prev_done = False
            prev_reward = reward
            obs = next_obs
            score += reward
        return encodings

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


if __name__ == "__main__":
    encoder = sma2c_encoder((5 + 2 + 1))
    # encoder.summary()
    ins = np.zeros(8)
    ins[3] = 1.0
    ins = ins[np.newaxis, :]
    # print(encoder(ins))
    actor, critic = sma2c_actor_critic(encoder, (5), (2))
    obs = np.zeros(5)
    obs[-1] = 1.
    obs = obs[np.newaxis, :]
    print(actor([ins, obs]))
