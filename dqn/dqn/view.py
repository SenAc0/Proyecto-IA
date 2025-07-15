import torch
from neural import MarioNet
from preprocessing import PreprocessFrame, FrameStack
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT  

# Cargar en CUDA si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear entorno con preprocesamiento
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = PreprocessFrame(env)
env = FrameStack(env, 4)

# Determinar dimensiones
n_actions = env.action_space.n
obs_shape = env.observation_space.shape  
input_dim = (obs_shape[2], obs_shape[0], obs_shape[1])  

# Cargar red entrenada
policy_net = MarioNet(input_dim, n_actions).to(device)
policy_net.load_state_dict(torch.load("mario_policy_net_episode_1000.pth"))
policy_net.eval()

state = env.reset()
done = False

while not done:
    env.render()
    state_tensor = torch.FloatTensor(state / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)  
    
    with torch.no_grad():
        q_values = policy_net(state_tensor, model='online')  
    action = q_values.argmax().item()

    next_state, reward, done, info = env.step(action)

    state = next_state

env.close()
