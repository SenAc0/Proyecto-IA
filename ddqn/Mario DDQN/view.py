import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from agente import Agent
import time

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear entorno
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = apply_wrappers(env)

# Inicializar el agente
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
agent.online.load_state_dict(torch.load("mario_policy_net_episode_1750.pth", map_location=device))
agent.online.to(device)
agent.online.eval()

# Jugar
state = env.reset()
done = False

while not done:
    env.render()
    time.sleep(1/30) 

    # Preprocesar estado para la red
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = agent.online(state_tensor)
    action = q_values.argmax().item()

    next_state, reward, done, info = env.step(action)
    state = next_state

env.close()
