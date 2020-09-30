import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

class ProbDist(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(tf.random.categorical(logits, 1), axis = -1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbDist()
        
    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)
    
    def action_value(self, obs):
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value,axis = -1)
    
import gym

env = gym.make('CartPole-v0')
model = Model(num_actions=env.action_space.n)

obs = env.reset()
# No feed_dict or tf.Session() needed at all!
action, value = model.action_value(obs[None, :])
print(action, value)