from gym.envs.registration import register

register(
    id='DB-v0',
    entry_point='gym_dbenv.envs:DBENV'
)
