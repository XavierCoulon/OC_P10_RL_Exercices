import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================
total_episodes = 15000
learning_rate = 0.8
discount_factor = 0.95

# Exploration
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.0005

# ==========================================
# 2. INITIALISATION
# ==========================================
print("--- Démarrage de l'entraînement (Mode Difficile / Glissant) ---")

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode=None)

# 16 états x 4 actions
q_table = np.zeros((env.observation_space.n, env.action_space.n)) # type: ignore

# Variable pour compter les victoires
success_count = 0

# ==========================================
# 3. BOUCLE D'ENTRAÎNEMENT
# ==========================================
for episode in range(total_episodes):

    state, info = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:

        # A. Epsilon-Greedy
        random_number = random.uniform(0, 1)
        if random_number < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        # B. Action
        new_state, reward, terminated, truncated, info = env.step(action)

        # C. Bellman
        max_future_q = np.max(q_table[new_state, :])
        current_q = q_table[state, action]

        new_q = current_q + learning_rate * (
            float(reward) + discount_factor * max_future_q - current_q
        )
        q_table[state, action] = new_q

        # D. Compteur de victoire
        # Dans FrozenLake, reward=1 signifie qu'on a atteint le cadeau
        if reward == 1:
            success_count += 1

        state = new_state

    # E. Decay Epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    # Petit log pour suivre la progression tous les 1000 épisodes
    if (episode + 1) % 1000 == 0:
        print(
            f"Épisode {episode + 1} | Taux de victoire actuel : {success_count / (episode + 1):.2%}"
        )

env.close()

# Résultat final
success_rate = (success_count / total_episodes) * 100
print(f"\n--- Entraînement terminé ---")
print(f"Nombre de victoires : {success_count} / {total_episodes}")
print(f"Taux de réussite global : {success_rate:.2f}%")


# ==========================================
# 4. VISUALISATION DE LA Q-TABLE (HEATMAP)
# ==========================================
def plot_q_table(q_table):
    """Affiche la Q-table sous forme de carte de chaleur"""
    plt.figure(figsize=(10, 6))

    # On utilise Seaborn pour faire une belle heatmap
    # Annot=True affiche les chiffres, fmt=".2f" arrondit à 2 décimales
    sns.heatmap(q_table, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)

    # Ajout des labels pour comprendre
    plt.title("Q-Table : Valeur de chaque action pour chaque état")
    plt.xlabel("Actions (0=Gauche, 1=Bas, 2=Droite, 3=Haut)")
    plt.ylabel("États (Cases du lac 0 à 15)")

    # On remplace les chiffres 0,1,2,3 par les noms des actions
    actions_labels = ["Gauche", "Bas", "Droite", "Haut"]
    plt.xticks(np.arange(4) + 0.5, actions_labels, rotation=0)

    plt.show()


print("\nAffichage de la Heatmap...")
plot_q_table(q_table)


# ==========================================
# 5. DÉMO FINALE
# ==========================================
print("\n--- Démarrage de la Démo Visuelle ---")
env_visu = gym.make(
    "FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human"
)

for episode in range(5):
    state, _ = env_visu.reset()
    terminated = False
    truncated = False
    print(f"Démo Épisode {episode + 1}")

    while not terminated and not truncated:
        action = np.argmax(q_table[state, :])
        state, reward, terminated, truncated, _ = env_visu.step(action)
        time.sleep(0.5)

env_visu.close()
