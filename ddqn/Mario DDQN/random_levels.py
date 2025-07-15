import os
import gym_super_mario_bros
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from agente import Agent

import random

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

#preguntar a usuario si ya conoce el device
respuesta = input("¿Quieres continuar con este dispositivo? (y/n): ").strip().lower()
if respuesta not in ['y', 'yes']:
    print("Programa terminado por el usuario.")
    exit()


# Lista de niveles permitidos
niveles = [
    'SuperMarioBros-8-1-v0',
    #'SuperMarioBros-1-3-v0',
    #'SuperMarioBros-2-1-v0',
    #'SuperMarioBros-4-1-v0',
    #'SuperMarioBros-5-1-v0'
]

# Factores de normalización de recompensas por nivel
reward_factors = {
    'SuperMarioBros-8-1-v0': 1.0,  
    #'SuperMarioBros-1-3-v0': 1.0,   
    #'SuperMarioBros-2-1-v0': 1.0,     
    #'SuperMarioBros-4-1-v0': 0.6,  
    #'SuperMarioBros-5-1-v0': 1.2    
}


#Variables para guardar mejor resultado.
reward11 = float('-inf')
reward13 = float('-inf')
reward21 = float('-inf')
reward81 = float('-inf')

env_temp = gym_super_mario_bros.make(niveles[0])
env_temp = JoypadSpace(env_temp, SIMPLE_MOVEMENT)
env_temp = apply_wrappers(env_temp)

# Crear agente y asegurar que use GPU si está disponible
agent = Agent(input_dims=env_temp.observation_space.shape, num_actions=env_temp.action_space.n, device=device)

# Verificar que las redes estén en GPU
if torch.cuda.is_available():
    print(" Redes neurales configuradas para GPU")
    print(f"   - Red online en: {next(agent.online.parameters()).device}")
    print(f"   - Red target en: {next(agent.target.parameters()).device}")
else:
    print(" Usando CPU para entrenamiento")

env_temp.close()

episodes = 30000  
save_interval = 2000
checkpoint_path = "mario_random_checkpoint_latest.pth"



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
    print(f"Checkpoint cargado desde el episodio {start_episode} en {device}")
else:
    print(f"No se encontró checkpoint, empezando desde el episodio 0")
    
print(f" Entrenamiento iniciado en {device}")


# Entrenamiento
for episode in range(start_episode, episodes):
    nivel_seleccionado = random.choice(niveles)
    env = gym_super_mario_bros.make(nivel_seleccionado)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = apply_wrappers(env)
    done = False
    state = env.reset()

    # Inicializar la recompensa total del episodio
    total_reward = 0  
    
    while not done:
        action = agent.elegir_accion(state)
        new_state, reward, done, _ = env.step(action)

        # Aplicar factor de normalización de recompensa por nivel
        reward = reward * reward_factors[nivel_seleccionado]

        total_reward += reward  # Acumular la recompensa
        
        agent.store_in_memory(state, action, reward, new_state, done)
        agent.learn()
        state = new_state
    
    env.close()
    
    # Limpiar cache de GPU para evitar problemas de memoria
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    if (episode + 1) % save_interval == 0:
        save_path = f"mario_policy_random_net_episode_{episode + 1}.pth"
        torch.save(agent.online.state_dict(), save_path)
        print(f"Modelo guardado en {save_path}")
        
    if (episode >= episodes - 10):
        save_path = f"mario_policy_random_net_episode_{episode + 1}.pth"
        torch.save(agent.online.state_dict(), save_path)
        print(f"Modelo guardado en {save_path}")
        
    # Guardar el mejor modelo por nivel
    if nivel_seleccionado == 'SuperMarioBros-8-1-v0' and total_reward > reward11:
        reward11 = total_reward
        torch.save(agent.online.state_dict(), "best_8-1.pth")
        print(f"🌟 Nuevo mejor modelo para 8-1 con recompensa {reward11:.2f}")
    
    if nivel_seleccionado == 'SuperMarioBros-1-1-v0' and total_reward > reward11:
        reward11 = total_reward
        torch.save(agent.online.state_dict(), "best_1-1.pth")
        print(f"🌟 Nuevo mejor modelo para 1-1 con recompensa {reward11:.2f}")

    elif nivel_seleccionado == 'SuperMarioBros-1-3-v0' and total_reward > reward13:
        reward13 = total_reward
        torch.save(agent.online.state_dict(), "best_1-3.pth")
        print(f"🌟 Nuevo mejor modelo para 1-3 con recompensa {reward13:.2f}")

    elif nivel_seleccionado == 'SuperMarioBros-2-1-v0' and total_reward > reward21:
        reward21 = total_reward
        torch.save(agent.online.state_dict(), "best_2-1.pth")
        print(f"🌟 Nuevo mejor modelo para 2-1 con recompensa {reward21:.2f}")
     
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
        log_file.write(f"Episodio: {episode + 1}, Nivel: {nivel_seleccionado}, Epsilon: {agent.epsilon}, Recompensa: {total_reward:.2f}\n")
    print(f"Episodio: {episode + 1}, Nivel: {nivel_seleccionado}, Epsilon: {agent.epsilon}, Recompensa: {total_reward:.2f}\n")

#env.close()
