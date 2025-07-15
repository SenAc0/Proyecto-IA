import os
import torch
import random
import numpy as np
from torch import optim
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from neural import MarioNet
import torch.nn as nn
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class FastMarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calcular tamaño 
        conv_out_size = self._get_conv_out(conv_layers, input_dim)
        
        # Red completa
        self.online = nn.Sequential(
            conv_layers,
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        import copy
        self.target = copy.deepcopy(self.online)
        
        for p in self.target.parameters():
            p.requires_grad = False
    
    def _get_conv_out(self, conv_layers, input_shape):
        """Calcular el tamaño de salida de las capas convolucionales"""
        o = conv_layers(torch.zeros(1, *input_shape))
        return int(np.prod(o.size()))
    
    def forward(self, input, model):
        if input.dtype != torch.float32:
            input = input.float()
        
        # Solo normalizar si no está normalizado
        if input.max() > 1.0:
            input = input / 255.0
            
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)
        else:
            raise ValueError(f"model debe ser 'online' o 'target', recibido: {model}")
    
    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

class SkipFrame:
    def __init__(self, env, skip):
        self.env = env
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self):
        return self.env.reset()

    def __getattr__(self, name):
        return getattr(self.env, name)

print("Iniciando el entrenamiento de Mario...")

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Información adicional sobre GPU si está disponible
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"   - Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   - Memoria libre: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
else:
    print("No se detectó GPU compatible, usando CPU")

# Preguntar a usuario si ya conoce el device
respuesta = input("¿Quieres continuar con este dispositivo? (y/n): ").strip().lower()
if respuesta not in ['y', 'yes']:
    print("Programa terminado por el usuario.")
    exit()



#Ya no se usa el preprocessing.py, ahora se aplica desde aqui directamente, simplemente por comodidad, mejor usar estas librerias existentes.

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)  
env = ResizeObservation(env, shape=(84, 84))
env = GrayScaleObservation(env, keep_dim=False)
env = FrameStack(env, num_stack=4, lz4_compress=True)

# Parámetros 
BUFFER_SIZE = 100_000
BATCH_SIZE = 32
GAMMA = 0.9              
EPS_START = 1.0
EPS_END = 0.05           
EPS_DECAY = 0.999998    
SYNC_NETWORK_RATE = 10_000  
LR = 0.00025            

episodes = 15000
save_interval = 1000
checkpoint_path = "mario_checkpoint_latest.pth"

n_actions = env.action_space.n
obs_shape = env.observation_space.shape
print(f"Forma de observación: {obs_shape}")

input_dim = obs_shape

# Red neuronal 
mario_net = FastMarioNet(input_dim, n_actions).to(device)
optimizer = optim.Adam(mario_net.online.parameters(), lr=LR)

storage = LazyMemmapStorage(BUFFER_SIZE)
replay_buffer = TensorDictReplayBuffer(storage=storage)

# Verificar que el modelo esté en GPU
if torch.cuda.is_available():
    print(f"Modelo cargado en GPU: {next(mario_net.parameters()).device}")
    print(f"Red online en: {next(mario_net.online.parameters()).device}")
    print(f"Red target en: {next(mario_net.target.parameters()).device}")
else:
    print("Modelo en CPU")

epsilon = EPS_START
learn_step_counter = 0

start_episode = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mario_net.online.load_state_dict(checkpoint['online_state_dict'])
    mario_net.target.load_state_dict(checkpoint['target_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epsilon = checkpoint['epsilon']
    learn_step_counter = checkpoint.get('learn_step_counter', 0)
    start_episode = checkpoint['episode']
    print(f"Checkpoint cargado desde el episodio {start_episode}")

def select_action(state, mario_net, epsilon, n_actions, device):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    observation = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = mario_net(observation, model='online')
    return q_values.argmax().item()

def store_in_memory(replay_buffer, state, action, reward, next_state, done):
    """Almacenar usando mismo formato que DDQN"""
    replay_buffer.add(TensorDict({
        'state': torch.tensor(np.array(state), dtype=torch.float32),
        'action': torch.tensor(action),
        'reward': torch.tensor(reward),
        'next_state': torch.tensor(np.array(next_state), dtype=torch.float32),
        'done': torch.tensor(bool(done), dtype=torch.bool)  
    }))

def optimize_model(mario_net, replay_buffer, optimizer, batch_size, gamma, device, learn_step_counter, epsilon, eps_end, eps_decay, sync_rate):
    if len(replay_buffer) < batch_size:
        return learn_step_counter, epsilon

    if learn_step_counter % sync_rate == 0 and learn_step_counter > 0:
        mario_net.update_target()

    optimizer.zero_grad()
    
    # Muestrear 
    samples = replay_buffer.sample(batch_size).to(device)
    
    # Procesar datos 
    states = samples['state'].float()
    actions = samples['action']
    rewards = samples['reward'].float()
    next_states = samples['next_state'].float()
    dones = samples['done'].float()

    # Calcular Q-values 
    predicted_q_values = mario_net(states, model='online')
    predicted_q_values = predicted_q_values[np.arange(batch_size), actions.squeeze()]
    
    target_q_values = mario_net(next_states, model='target').max(dim=1)[0]
    target_q_values = rewards + gamma * target_q_values * (1 - dones.float())

    # Usar MSE loss 
    loss = torch.nn.functional.mse_loss(predicted_q_values, target_q_values)
    loss.backward()
    optimizer.step()
    
    # Actualizar contadores y epsilon 
    learn_step_counter += 1
    epsilon = max(eps_end, epsilon * eps_decay)
    
    return learn_step_counter, epsilon

print("-" * 60)

# Entrenamiento
for episode in range(start_episode, episodes):
    done = False
    state = env.reset()
    total_reward = 0

    # Verificación de GPU en el primer episodio (por si acaso, no quitar)
    if episode == start_episode and torch.cuda.is_available():
        print(f"🔍 Verificando uso de GPU en episodio {episode + 1}...")

    while not done:
        action = select_action(state, mario_net, epsilon, n_actions, device)
        next_state, reward, done, info = env.step(action)
        
        # Guardar en memoria 
        store_in_memory(replay_buffer, state, action, reward, next_state, done)
        
        # Optimizar modelo después de cada step 
        learn_step_counter, epsilon = optimize_model(
            mario_net, replay_buffer, optimizer, BATCH_SIZE, GAMMA, device,
            learn_step_counter, epsilon, EPS_END, EPS_DECAY, SYNC_NETWORK_RATE
        )
        
        state = next_state
        total_reward += reward

    # Progreso cada episodio con reward y epsilon
    print(f"Episodio {episode + 1} | Reward: {total_reward:.2f} | ε: {epsilon:.3f}")

    # Guardar en txt episode, reward y epsilon
    with open("mario_training_log.txt", "a") as log_file:
        log_file.write(f"Episodio: {episode + 1}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}\n")

    # Guardar modelo cada save_interval episodios
    if (episode + 1) % save_interval == 0:
        save_path = f"mario_policy_net_episode_{episode + 1}.pth"
        torch.save(mario_net.online.state_dict(), save_path)
        print(f"Modelo guardado en {save_path}")

    # Guardar checkpoint cada N episodios
    if (episode + 1) % save_interval == 0:
        checkpoint = {
            'episode': episode + 1,
            'online_state_dict': mario_net.online.state_dict(),
            'target_state_dict': mario_net.target.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epsilon': epsilon,
            'learn_step_counter': learn_step_counter
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint guardado en {checkpoint_path} (Episodio {episode + 1})")

env.close()
print("\n¡Entrenamiento DQN completado!")
print(f"Episodios entrenados: {episodes}")
print(f"Learn steps realizados: {learn_step_counter}")
print(f"Epsilon final: {epsilon:.4f}")
