import torch
import torch.nn as nn
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from torchvision import transforms as T
import numpy as np
import gym
from pathlib import Path
import os

# === Custom wrappers ===
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)
    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        return torch.tensor(observation.copy(), dtype=torch.float)
    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        return transform(observation)

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = shape
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)
    def observation(self, observation):
        transforms = T.Compose([
            T.Resize(self.shape, antialias=True),
            T.Normalize(0, 255)
        ])
        return transforms(observation).squeeze(0)

# === Your trained network class ===
class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        self.online = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.target = nn.Sequential(*self.online)

    def forward(self, x):
        return self.online(x)

# === Auto-load latest checkpoint ===
def get_latest_checkpoint(base_path="checkpoints"):
    base_path = Path(base_path)
    checkpoint_dirs = sorted(base_path.glob("*/"), key=os.path.getmtime, reverse=True)
    if not checkpoint_dirs:
        raise FileNotFoundError("No checkpoint directories found.")
    latest_dir = checkpoint_dirs[0]
    chkpt_files = sorted(latest_dir.glob("*.chkpt"), key=os.path.getmtime, reverse=True)
    if not chkpt_files:
        raise FileNotFoundError(f"No .chkpt files found in {latest_dir}")
    print(f"[✅] Loading checkpoint: {chkpt_files[0]}")
    return chkpt_files[0]

# === Build environment ===
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

# === Load model ===
input_dim = (4, 84, 84)
n_actions = env.action_space.n
model = MarioNet(input_dim, n_actions)
checkpoint_path = get_latest_checkpoint("checkpoints")
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model"])
model.eval()

# === Watch Mario play ===
state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
while True:
    with torch.no_grad():
        state_tensor = torch.tensor(state.__array__(), dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()
    next_state, reward, done, trunc, info = env.step(action)
    state = next_state
    if done or info.get("flag_get"):
        state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
