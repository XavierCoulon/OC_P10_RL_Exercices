import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random


class DQN(nn.Module):
    """
    Réseau de neurones pour Deep Q-Learning (DQN).
    Hérite de torch.nn.Module pour intégration avec PyTorch.
    """

    def __init__(self, state_size, action_size, hidden_size=64):
        """
        Initialise le réseau DQN avec des couches linéaires.

        Args:
            state_size (int): Dimension de l'espace d'états
            action_size (int): Nombre d'actions possibles
            hidden_size (int): Taille des couches cachées (par défaut 64)
        """
        super(DQN, self).__init__()

        # Définition des couches du réseau
        self.fc1 = nn.Linear(state_size, hidden_size)  # Couche d'entrée
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Couche cachée
        self.fc3 = nn.Linear(hidden_size, action_size)  # Couche de sortie

    def forward(self, state):
        """
        Passage avant du réseau: propage l'état à travers les couches.

        Args:
            state (torch.Tensor): État d'entrée de forme (batch_size, state_size)

        Returns:
            torch.Tensor: Q-valeurs pour chaque action de forme (batch_size, action_size)
        """
        # Passage à travers la première couche avec ReLU
        x = F.relu(self.fc1(state))

        # Passage à travers la deuxième couche avec ReLU
        x = F.relu(self.fc2(x))

        # Passage à travers la couche de sortie (pas d'activation)
        q_values = self.fc3(x)

        return q_values


class ReplayBuffer:
    """
    Buffer de rejeu pour stocker et échantillonner les transitions
    dans l'apprentissage par renforcement.
    """

    def __init__(self, capacity=10000):
        """
        Initialise le buffer de rejeu avec une capacité fixe.

        Args:
            capacity (int): Nombre maximal de transitions à stocker (par défaut 10000)
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """
        Ajoute une transition au buffer de rejeu.

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done (bool): Indicateur de fin d'épisode
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Extrait un batch aléatoire du buffer de rejeu.

        Args:
            batch_size (int): Taille du batch à extraire

        Returns:
            list: Liste de tuples (state, action, reward, next_state, done)

        Raises:
            ValueError: Si batch_size est supérieur à la taille du buffer
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"batch_size ({batch_size}) ne peut pas dépasser "
                f"la taille du buffer ({len(self.buffer)})"
            )
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Retourne le nombre de transitions actuellement stockées."""
        return len(self.buffer)


# Exemple d'utilisation
if __name__ == "__main__":
    # Paramètres
    state_size = 4
    action_size = 2
    batch_size = 32
    capacity = 1000

    # Création du réseau DQN
    print("=" * 50)
    print("Création du réseau DQN")
    print("=" * 50)
    dqn = DQN(state_size, action_size)
    print(f"Réseau DQN créé:\n{dqn}\n")

    # Création du buffer de rejeu
    print("=" * 50)
    print("Création du ReplayBuffer")
    print("=" * 50)
    replay_buffer = ReplayBuffer(capacity=capacity)
    print(f"ReplayBuffer créé avec une capacité de {capacity}\n")

    # Ajout de quelques transitions au buffer
    print("=" * 50)
    print("Ajout de transitions au buffer")
    print("=" * 50)
    for i in range(50):
        state = torch.randn(state_size)
        action = random.randint(0, action_size - 1)
        reward = random.uniform(-1, 1)
        next_state = torch.randn(state_size)
        done = random.choice([True, False])

        replay_buffer.push(state, action, reward, next_state, done)

    print(f"Nombre de transitions dans le buffer: {len(replay_buffer)}\n")

    # Échantillonnage d'un batch
    print("=" * 50)
    print("Échantillonnage d'un batch")
    print("=" * 50)
    batch = replay_buffer.sample(batch_size=32)
    print(f"Batch de taille {len(batch)} extrait avec succès\n")

    # Test du forward pass du réseau
    print("=" * 50)
    print("Test du forward pass du réseau DQN")
    print("=" * 50)
    test_states = torch.randn(batch_size, state_size)
    q_values = dqn(test_states)
    print(f"État d'entrée: {test_states.shape}")
    print(f"Q-valeurs de sortie: {q_values.shape}")
    print(f"Q-valeurs (premiers échantillons):\n{q_values[:5]}")
