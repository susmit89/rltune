import gym
import gym_dbenv


env = gym.make('DB-v0')
print env.action_space.high
print env.action_space.low
print env.observation_space
print env.action_space
