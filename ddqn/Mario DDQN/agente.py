import torch
import numpy as np
from agente_nn import AgentNN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class Agent:
    def __init__(self, input_dims, num_actions, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_actions = num_actions
        self.learn_step_counter = 0
        
        #Hiperparámetros
        self.lr = 0.00025
        self.gamma = 0.9
        self.epsilon = 1.0
        self.eps_decay_rate = 0.999999  
        self.eps_min = 0.05
        self.batch_size = 32
        self.sync_network_rate=10_000
        
        #Networks
        self.online = AgentNN(input_dims, num_actions).to(device)
        self.target = AgentNN(input_dims, num_actions, freeze=True).to(device)
        
        #Optimizer y funcion loss
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        
        #Replay buffer
        replay_buffer_cap = 100_000
        storage = LazyMemmapStorage(replay_buffer_cap)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)
        
    def elegir_accion(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        observation = torch.tensor(np.array(observation), dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.online(observation).argmax().item()
    
    def eps_decay(self):
        self.epsilon = max(self.epsilon * self.eps_decay_rate, self.eps_min)
            
    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
            'state': torch.tensor(np.array(state), dtype=torch.float32),
            'action': torch.tensor(action),
            'reward': torch.tensor(reward),
            'next_state': torch.tensor(np.array(next_state), dtype=torch.float32),
            'done': torch.tensor(done, dtype=torch.bool)
        }))
        
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target.load_state_dict(self.online.state_dict()) 
            
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.sync_networks()
        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size).to(self.online.device)
        keys = ['state', 'action', 'reward', 'next_state', 'done']

        states = samples['state'].float()
        actions = samples['action']
        rewards = samples['reward'].float()
        next_states = samples['next_state'].float()
        dones = samples['done'].float()

        predicted_q_values = self.online(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        best_actions = self.online(next_states).argmax(dim=1)
        target_q_values = self.target(next_states)[np.arange(self.batch_size), best_actions]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.loss_fn(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.eps_decay()
