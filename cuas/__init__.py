from gym.envs.registration import register

# Register our environment
register(
    id="cuas-v0",
    entry_point="cuas.envs:CuasEnv",
)

#register(
#    id="cuas_single_agent-v0",
#    entry_point="cuas.envs:CuasEnvSingleAgent",
#)

#register(
#    id="cuas_multi_agent-v0",
#    entry_point="cuas.envs:CuasEnvMultiAgent",
#)

#register(
#    id="cuas_multi_agent-v_8o5",
#    entry_point="cuas.envs:cuas_env_class",
#)

register(
    id="cuas_env_class-v1",
    entry_point="cuas.envs:cuas_env_class",
)

register(
    id="cuas_env_class-v2",
    entry_point="cuas.envs:cuas_env_classv2",
)

#register(
#    id="cuas_env_class_No_Latency-v1",
#    entry_point="cuas.envs:cuas_env_class_No_Latency",
#)
