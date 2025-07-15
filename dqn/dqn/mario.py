from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
from gym_super_mario_bros.actions import RIGHT_ONLY
from preprocessing import PreprocessFrame, FrameStack
import matplotlib.pyplot as plt

from neural import MarioNet
import torch

# Crear el entorno con las acciones limitadas
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)


print("Acciones SIMPLE_MOVEMENT:")
for i, action in enumerate(SIMPLE_MOVEMENT):
    print(f"{i}: {action}")
print("Número total de acciones:", env.action_space.n)

# Aplicar preprocesamiento y stacking de frames
env = PreprocessFrame(env)
env = FrameStack(env, 4)

# Probar que funciona
state = env.reset()
print("Shape del estado preprocesado:", state.shape)  




plt.imshow(state[:, :, 0], cmap='gray')
plt.title('Primer frame apilado')
plt.show()