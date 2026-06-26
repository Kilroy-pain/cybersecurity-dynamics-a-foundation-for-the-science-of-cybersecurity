import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class CyberSecurityDynamicsModel(nn.Module):
    def __init__(self, num_nodes, attack_prob, defense_prob):
        super(CyberSecurityDynamicsModel, self).__init__()
        self.num_nodes = num_nodes
        self.attack_prob = attack_prob
        self.defense_prob = defense_prob
        self.state = torch.zeros(num_nodes, dtype=torch.float32)  # 0: secure, 1: compromised

    def forward(self):
        # Simulate attack-defense dynamics
        attack_events = torch.bernoulli(torch.full((self.num_nodes,), self.attack_prob))
        defense_events = torch.bernoulli(torch.full((self.num_nodes,), self.defense_prob))
        
        # Update state: compromised nodes remain compromised unless defended
        self.state = torch.clamp(self.state + attack_events - defense_events, 0, 1)
        return self.state

def simulate_cybersecurity_dynamics(num_nodes, attack_prob, defense_prob, steps):
    model = CyberSecurityDynamicsModel(num_nodes, attack_prob, defense_prob)
    states_over_time = []

    for _ in range(steps):
        state = model()
        states_over_time.append(state.clone().numpy())

    return np.array(states_over_time)

if __name__ == '__main__':
    # Parameters
    num_nodes = 10  # Number of nodes in the network
    attack_prob = 0.3  # Probability of an attack on a node at each step
    defense_prob = 0.2  # Probability of a defense on a node at each step
    steps = 50  # Number of time steps to simulate

    # Run simulation
    states = simulate_cybersecurity_dynamics(num_nodes, attack_prob, defense_prob, steps)

    # Print results
    print("Cybersecurity Dynamics Simulation:")
    print("Time Step | Compromised Nodes")
    for t, state in enumerate(states):
        print(f"{t:9} | {int(state.sum())} compromised nodes")