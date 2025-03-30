import torch
from torch import nn
import numpy as np
from pathlib import Path
import datetime
import gym
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchvision import transforms as T

# Custom Wrappers
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)
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
        if isinstance(shape, int): shape = (shape, shape)
        self.shape = shape
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    def observation(self, observation):
        transforms = T.Compose([T.Resize(self.shape, antialias=True), T.Normalize(0, 255)])
        return transforms(observation).squeeze(0)

# Mario Agent
class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != 84 or w != 84:
            raise ValueError(f"Expecting input 84x84, got {h}x{w}")
        self.online = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.target = nn.Sequential(*self.online)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False
    def forward(self, input, model):
        return self.online(input) if model == "online" else self.target(input)

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = MarioNet(state_dim, action_dim).float().to(self.device)
        # Ensure online network parameters require gradients
        for p in self.net.online.parameters():
            p.requires_grad = True
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5e4
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device="cpu"))
        self.batch_size = 32
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = nn.SmoothL1Loss()
        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state.__array__(), dtype=torch.float32, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate * self.exploration_rate_decay)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        state = np.array(state.__array__(), dtype=np.float32)
        next_state = np.array(next_state.__array__(), dtype=np.float32)
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)
        self.memory.add(TensorDict({
            "state": state, "next_state": next_state, "action": action,
            "reward": reward, "done": done
        }, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")
        td_est = current_Q[torch.arange(self.batch_size, device=self.device), action]
        print(f"td_estimate - current_Q grad: {current_Q.requires_grad}, td_est grad: {td_est.requires_grad}")
        return td_est

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[torch.arange(self.batch_size, device=self.device), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        print(f"Loss requires grad: {loss.requires_grad}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin or self.curr_step % self.learn_every != 0:
            return None, None
        state, next_state, action, reward, done = self.recall()
        print(f"State grad: {state.requires_grad}, Action grad: {action.requires_grad}")
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss
    
# Setup Environment
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

# Training Loop
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
episodes = 40000  # Your setting
for e in range(episodes):
    reset_output = env.reset()
    state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
    while True:
        action = mario.act(state)
        step_output = env.step(action)
        next_state, reward, done, trunc, info = step_output
        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn()
        state = next_state
        if done or info["flag_get"]:
            break
    print(f"Episode {e} - Step {mario.curr_step} - Epsilon {mario.exploration_rate:.3f}")

# Visualization
print("Training complete. Now testing Mario visually...")
env.close()
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
for _ in range(1000):
    action = mario.act(state)
    next_state, reward, done, trunc, info = env.step(action)
    env.render()
    state = next_state
    if done or info["flag_get"]:
        state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
env.close()