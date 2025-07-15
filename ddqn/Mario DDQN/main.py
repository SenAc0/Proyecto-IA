import os
import gym_super_mario_bros
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from agente import Agent

print("Iniciando el entrenamiento de Mario...")

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = apply_wrappers(env)

episodes = 30000  
save_interval = 2000
checkpoint_path = "mario_checkpoint_latest.pth"

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Cargar desde checkpoint si existe
start_episode = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.online.load_state_dict(checkpoint['model_state_dict'])
    agent.target.load_state_dict(checkpoint['target_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    agent.learn_step_counter = checkpoint['learn_step_counter']
    start_episode = checkpoint['episode']
    print(f"Checkpoint cargado desde el episodio {start_episode}")


# Entrenamiento
for episode in range(start_episode, episodes):
    done = False
    state = env.reset()
    
    while not done:
        action = agent.elegir_accion(state)
        new_state, reward, done, _ = env.step(action)

        agent.store_in_memory(state, action, reward, new_state, done)
        agent.learn()
        state = new_state
        
    if (episode + 1) % save_interval == 0:
        save_path = f"mario_policy_net_episode_{episode + 1}.pth"
        torch.save(agent.online.state_dict(), save_path)
        print(f"Modelo guardado en {save_path}")

    # Guardar checkpoint cada N episodios
    if (episode + 1) % save_interval == 0:
        checkpoint = {
            'episode': episode + 1,
            'model_state_dict': agent.online.state_dict(),
            'target_state_dict': agent.target.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'learn_step_counter': agent.learn_step_counter
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint guardado en {checkpoint_path} (Episodio {episode + 1})")
    #Guardar en txt episode and reward
    with open("mario_training_log.txt", "a") as log_file:
        log_file.write(f"Episodio: {episode + 1}, Epsilon: {agent.epsilon:.4f}\n")

env.close()
