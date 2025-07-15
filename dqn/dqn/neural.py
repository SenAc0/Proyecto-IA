from torch import nn
import torch
import copy

class MarioNet(nn.Module):
    """Red neuronal para DQN con arquitectura mejorada."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        
        self.online = nn.Sequential(
            # Primera capa convolucional
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Segunda capa convolucional
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Tercera capa convolucional
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            
            # Aplanar
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(512, output_dim)
        )
        
        # Red target 
        self.target = copy.deepcopy(self.online)
        
        # La red target no se entrena
        for p in self.target.parameters():
            p.requires_grad = False
    
    def forward(self, input, model):
        """Forward pass através de la red especificada."""
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)
        else:
            raise ValueError(f"model debe ser 'online' o 'target', recibido: {model}")
    
    def update_target(self):
        """Actualiza la red target con los pesos de la red online."""
        self.target.load_state_dict(self.online.state_dict())
