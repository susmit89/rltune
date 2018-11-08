import gym
import gym_dbenv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.models import load_model
import rl
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory
from keras.models import load_model
ENV_NAME = 'DB-v0'
env = gym.make(ENV_NAME)
print env.observation_space
print env.observation_space.shape
print (1,) + env.observation_space.shape
print env.action_space
nb_actions = env.action_space.n
env.render()
try:
    model = load_model('db_model.h5')
    print("Using saved model")
except:
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(48))
    model.add(Activation('relu'))
    model.add(Dense(48))
    model.add(Activation('relu'))
    model.add(Dense(48))
    model.add(Activation('relu'))
    model.add(Dense(48))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('sigmoid'))
print(model.summary())
#memory = SequentialMemory(limit=50000, window_length=1)
memory = EpisodeParameterMemory(limit=2000, window_length=1)
#policy = BoltzmannQPolicy()
dqn = rl.agents.cem.CEMAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05
               )
dqn.compile()

dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)

dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
model.save('db_model.h5')
# Finally, evaluate our algorithm for 5 episodes.
#dqn.test(nb_episodes=10, visualize=True)
