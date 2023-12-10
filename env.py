import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from env import WarehouseEnvironment

class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.output_layer = Dense(action_dim, activation='tanh')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.output_layer = Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(action_dim)
        self.critic = Critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    def get_action(self, state):
        normalized_state = (state.flatten() - state.flatten().min()) / (state.flatten().max() - state.flatten().min())
        return self.actor(np.array([normalized_state])).numpy()

    def train(self, state, action, reward, next_state, done):
        normalized_state = (state.flatten() - state.flatten().min()) / (state.flatten().max() - state.flatten().min())
        normalized_next_state = (next_state.flatten() - next_state.flatten().min()) / (
                next_state.flatten().max() - next_state.flatten().min())

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            target_actions = self.actor(normalized_next_state)
            target_q = reward + 0.99 * (1 - done) * self.critic(normalized_next_state, target_actions)
            predicted_q = self.critic(normalized_state, action)
            critic_loss = tf.keras.losses.mean_squared_error(target_q, predicted_q)

        critic_gradients = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        actor_loss = -self.critic(normalized_state, self.actor(normalized_state))
        actor_gradients = tape1.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

# Example usage:
# env = WarehouseEnvironment(width=5, height=5)
# state_dim = env.observation_space.shape
# action_dim = env.action_space.n
# agent = DDPGAgent(state_dim, action_dim)
# for episode in range(1000):
#     state = env.reset()
#     episode_reward = 0
#     for step in range(200):
#         action = agent.get_action(state)
#         next_state, reward, done, _ = env.step(action)
#         agent.train(state, action, reward, next_state, done)
#         state = next_state
#         episode_reward += reward
#         if done:
#             break
#     print(f"Episode: {episode + 1}, Reward: {episode_reward}")
# env.close()
