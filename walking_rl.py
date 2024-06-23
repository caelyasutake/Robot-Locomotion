import numpy as np
import pybullet as p
import pybullet_data
import tensorflow as tf
from tensorflow.keras import layers
from scipy.stats import beta
import gym

class RobotEnv:
    def __init__(self, nq, q_req):
        self.nq = nq
        self.physics_client = p.connect(p.DIRECT)

        # setup environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -98.1)

        # setup robot
        robot_start_position = [0, 0, 0.1]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.setAdditionalSearchPath("/cad2")
        self.robot_id = p.loadURDF("shibot_inu.urdf", robot_start_position, robot_start_orientation)
        self.mode = p.POSITION_CONTROL

        self.state = self.get_state()

        # action parameter contraints
        self.delta_min = -2.0
        self.delta_max = 2.0

        # target position
        self.q_req = q_req

        # fall thresholds
        self.pz_min = 0.05
        self.ralpha_max = 100
        self.rbeta_max = 100

        # reward weight hyperparameters
        self.w_v = 1.0
        self.w_r = 1.0
        self.w_a = 1.0
        self.w_f = 1.0

    def get_state(self, v_des=np.zeros(3), v_cmd=np.zeros(3)):
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        R = np.array([roll, pitch, yaw])

        # get joint information
        joint_positions = []
        joint_velocities = []

        for joint_index in range(self.nq):
            joint_state = p.getJointState(self.robot_id, joint_index)

            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])

        joint_positions = np.array(joint_positions)
        joint_velocities = np.array(joint_velocities)

        # set up inertial reference frame with base velocity
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_id)
        I = p.getMatrixFromQuaternion(orientation)
        I = np.array(I).reshape(3, 3)

        linear_velocity_base_link = I.T @ linear_velocity

        return np.array([joint_positions,
                         joint_velocities,
                         R,
                         angular_velocity,
                         linear_velocity_base_link,
                         linear_velocity,
                         v_des,
                         v_cmd])

    def sample_action(self):
        # define beta distribution
        alpha = 2.0
        beta_param = 5.0

        sampling_rate = 40 # Hz
        sampling_interval = 1.0 / sampling_rate

        delta = beta(alpha, beta_param).rvs()
        delta = self.delta_min + (self.delta_max - self.delta_min) * delta

        return delta

    def logistic_kernel(self, x, l=1):
        return 2 / (np.exp(l * x) + np.exp(-l * x))

    def calculate_reward(self, delta_time):
        '''
        reward terms bounded such that r_* in [0, 1] * delta_time
        delta_time is the time stpe of the policy controller

        smooth logistic kernel function used to bound reward terms
        K:R -> [0, 1]
        K(x|l) = 2/(e^(lx) + e^(-lx))
        l: sensitivity of the kernel

        :return: reward
        '''
        joint_positions, joint_velocities, R, angular_velocity, linear_velocity_base_link, linear_velocity, v_des, v_cmd = self.state

        # velocity tracking
        r_vel = np.linalg.norm(np.array([v_cmd[0] - linear_velocity[0],
                                         v_cmd[1] - linear_velocity[1],
                                         v_cmd[0] - linear_velocity_base_link[0],
                                         v_cmd[1] - linear_velocity_base_link[1],
                                         v_cmd[2] - angular_velocity[2]]))

        # pose regularization
        r_reg = self.logistic_kernel(np.linalg.norm(self.q_req - joint_positions)) * delta_time

        # alive
        position, orientation = p.getBaseVelocity(self.robot_id)

        r_alive = 0
        if position[2] > self.pz_min and orientation[0] < self.ralpha_max and orientation[1] < self.rbeta_max:
            r_alive = delta_time

        # foot clearance
        #e_f, r_foot

        return self.w_v * r_vel + self.w_r * r_reg + self.w_a * r_alive # + self.w_f * r_foot

class PPO_Agent:
    def __init__(self, nq, actor_lr=0.0003, critic_lr=0.0003):
        self.nq = nq
        self.actor = self.build_actor(nq)
        self.critic = self.build_critic(nq)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.gamma = 0.99
        self.lam = 0.95
        self.epsilon = 0.2

    def build_actor(self, nq):
        state_dim = np + 16
        inputs = tf.keras.Input(shape=(state_dim,))
        x = layers.Dense(512, activation='tanh')(inputs)
        x = layers.Dense(512, activation='tanh')(x)
        outputs = layers.Dense(nq * 2, activation='tanh')(x)
        return tf.keras.Model(inputs, outputs)

    def build_critic(self, nq):
        state_dim = nq + 16
        inputs = tf.keras.Input(shape=(state_dim,))
        x = layers.Dense(512, activation='tanh')(inputs)
        x = layers.Dense(512, activation='tanh')(x)
        outputs = layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs)

    def select_action(self, state):
        state = np.expand_dims(state, axis=0)
        mu, sigma = np.split(self.actor(state), 2, axis=-1)
        sigma = tf.nn.softplus(sigma) + 1e-5
        dist = tf.compat.v1.distributions.Normal(mu, sigma)
        action = dist.sample()
        return action[0].numpy()

    def train(self, states, actions, advantages, returns):
        with tf.GradientTape() as tape:
            values = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(returns - values))

        grads = tape.gradient(critic_loss, self.critic_trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            mu, sigma = tf.split(self.actor(states), 2, axis=-1)
            sigma = tf.nn.softplus(sigma) + 1e-5
            dist = tf.compat.v1.distributions.Normal(mu, sigma)
            log_probs = dist.log_prob(actions)
            ratio = tf.exp(log_probs - tf.stop_gradient(log_probs))
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        grads = tape.gradient(actor_loss, self.actor_trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def compute_gae(self, rewards, values, next_values, dones, gamma, lam):
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = lastgaelam = delta + gamma * lam * (1 - dones[t])
        returns = advantages + values
        return advantages, returns

def train_agent(env, agent, num_epochs=1000):
    for epoch in range(num_epochs):
        state = env.reset()
        states, actions, rewards, dones, next_states = [], [], [], [], []

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            state = next_state

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        next_states = np.array(next_states)

        values = agent.critic(states).numpy().flatten()
        next_values = agent.critic(next_states).numpy().flatten()
        advantages, returns = agent.compute_gae(rewards, values, next_values, dones, agent.gamma, agent.lam)

        agent.train(states, actions, advantages, returns)

nq = 8
q_env = 0
env = RobotEnv()
nq = 8
agent = PPOAgent(nq)
train_agent(env, agent, num_epochs=1000)

'''
class State:
    def __init__(self, nq):
        self.nq = nq
        self.q = np.zeros(nq) # joint positions
        self.q_dot = np.zeros(nq) # joint velocities
        self.R = np.zeros(3) # orientation of robot base
        self.omega_b = np.zeros(3) # velocity of base w.r.t a inertial reference frame
        self.IvIb = np.zeros(2)
        self.bvIb = np.zeros(2)
        self.v_des = np.zeros(3) # long term desired velocity
        self.v_cmd = np.zeros(3) # short term commanded velocity

class Action:
    def __init__(self, nq):
        self.delta = np.zeros(nq)

def build_actor(nq):
    state_dim = np + 16
    inputs = tf.keras.Input(shape=(state_dim,))
    x = layers.Dense(512, activation='tanh')(inputs)
    x = layers.Dense(512, activation='tanh')(x)
    outputs = layers.Dense(nq * 2, activation='tanh')(x)
    return tf.keras.Model(inputs, outputs)

def build_critic(nq):
    state_dim = nq + 16
    inputs = tf.keras.Input(shape=(state_dim,))
    x = layers.Dense(512, activation='tanh')(inputs)
    x = layers.Dense(512, activation='tanh')(x)
    outputs = layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)

class PPOAgent:
    def __init__(self, nq, actor_lr=0.0003, critic_lr=0.0003):
        self.nq = nq
        self.actor = build_actor(nq)
        self.critic = build_critic(nq)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.gamma = 0.99
        self.lam = 0.95
        self.epsilon = 0.2

    def select_action(self, state):
        state = np.expand_dims(state, axis=0)
        mu, sigma = np.split(self.actor(state), 2, axis=-1)
        sigma = tf.nn.softplus(sigma) + 1e-5
        dist = tf.compat.v1.distributions.Normal(mu, sigma)
        action = dist.sample()
        return action[0].numpy()

    def train(self, states, actions, advantages, returns):
        with tf.GradientTape() as tape:
            values = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(returns - values))

        grads = tape.gradient(critic_loss, self.critic_trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            mu, sigma = tf.split(self.actor(states), 2, axis=-1)
            sigma = tf.nn.softplus(sigma) + 1e-5
            dist = tf.compat.v1.distributions.Normal(mu, sigma)
            log_probs = dist.log_prob(actions)
            ratio = tf.exp(log_probs - tf.stop_gradient(log_probs))
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        grads = tape.gradient(actor_loss, self.actor_trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

def compute_gae(rewards, values, next_values, dones, gamma, lam):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * (1 - dones[t])
    returns = advantages + values
    return advantages, returns

def train_agent(env, agent, num_epochs=1000):
    for epoch in range(num_epochs):
        state = env.reset()
        states, actions, rewards, dones, next_states = [], [], [], [], []

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            state = next_state

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        next_states = np.array(next_states)

        values = agent.critic(states).numpy().flatten()
        next_values = agent.critic(next_states).numpy().flatten()
        advantages, returns = compute_gae(rewards, values, next_values, dones, agent.gamma, agent.lam)

        agent.train(states, actions, advantages, returns)

def compute_reward(state, action, next_state):
    r_vel = -np.linalg.norm(state.v_cmd - state.bvIb)
    r_pos = -np.linalg.norm(state.q - action.delta)
    r_alive = 1.0 if state.q[2] > 0.05 else 0.0
    r_foot_clearance = 1.0 if np.all(state.q_dot > 0.05) else 0.0

    reward = r_vel + r_pos + r_alive + r_foot_clearance
    return reward

env = gym.make('QuadrupedRobot-v0')
nq = 8
agent = PPOAgent(nq)
train_agent(env, agent, num_epochs=1000)

'''