import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """Buffer de experiencia mejorado para DQN."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Almacena una transición en el buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        self.position = (self.position + 1) % self.buffer.maxlen
    
    def sample(self, batch_size):
        """Muestrea un batch aleatorio de transiciones."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer contiene {len(self.buffer)} elementos, pero se requieren {batch_size}")
            
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def can_sample(self, batch_size):
        """Verifica si se puede muestrear un batch del tamaño especificado."""
        return len(self.buffer) >= batch_size
