import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
before_training = "video/before_training.mp4"

env = gym.make("CartPole-v1", render_mode="rgb_array")
video = VideoRecorder(env, before_training)
observation, info = env.reset(seed=42)

for i in range(1000):
    env.render()
    video.capture_frame()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # print("step", i, observation, reward, terminated, truncated, info)
    if terminated or truncated:
        observation, info = env.reset()
video.close()
env.close()