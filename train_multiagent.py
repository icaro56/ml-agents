from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO2, PPO1
from stable_baselines.a2c import A2C
from UnityVecEnv import UnityVecEnv
import os
import psutil


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[41, 41, 41],
                                           feature_extraction="mlp")

def main():

    env_folder = "unity-volume"
    env_id = "bomberman"
    # env_id = "walker"

    env_path = ""
    if psutil.LINUX:
        env_path = os.path.join('envs', "linux", env_id, "Unity Environment.x86_64")
    elif psutil.WINDOWS:
        env_path = os.path.join(env_folder, "windows", env_id, 'Bomberman.exe')

    env = UnityVecEnv(os.path.join(env_path))
    # env = UnityVecEnv(os.path.join('marathon_envs', 'Walker'))
    # env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)

    # Automatically normalize the input features
    env = VecNormalize(env)
    # env = VecNormalize(env, norm_obs=True, norm_reward=False,clip_obs=10.)
    # env = VecNormalize(env, norm_obs=True, norm_reward=False)

    tensorboard_log = os.path.join("summaries", env_id)
    os.makedirs(tensorboard_log, exist_ok=True)
    policy = MlpPolicy
    # policy = MlpLstmPolicy
    # policy = MlpLnLstmPolicy
    # policy = CustomPolicy
    # model = PPO2(policy, env,
    #     gamma=0.99,
    #     learning_rate=1.0e-3,
    #     lam=0.95,
    #     n_steps=512,
    #     verbose=2, tensorboard_log=tensorboard_log
    #     )
    # model = PPO2(policy=policy, env=env, n_steps=2048, nminibatches=32, lam=0.95, gamma=0.99, noptepochs=10,
    # model = PPO2(policy=policy, env=env, n_steps=10240, nminibatches=2048, lam=0.95, gamma=0.99, noptepochs=3,
    # model = PPO2(policy=policy, env=env, n_steps=640, nminibatches=2048, lam=0.95, gamma=0.99, noptepochs=3,
    # model = PPO2(policy=policy, env=env, n_steps=64, nminibatches=16, lam=0.95, gamma=0.99, noptepochs=8,
    model = PPO2(policy=policy, env=env, 
                    # n_steps=64, nminibatches=16, lam=0.95, gamma=0.99, noptepochs=8,
                    # ent_coef=0.0, learning_rate=1e-3, 
                    # ent_coef=0.0, 
                    # learning_rate=3e-4,
                    # cliprange=0.2,
                    verbose=2, tensorboard_log=tensorboard_log
                    )
    # model = A2C(policy, env,
    #     verbose=2, tensorboard_log=tensorboard_log
    #     )
    timesteps = 1000
    model.learn(total_timesteps=timesteps)
    os.makedirs('models', exist_ok=True)
    model.save(os.path.join("models", env_id))


if __name__ == '__main__':
    main()
