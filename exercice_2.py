import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================
total_episodes = 15000  # Suffisant pour converger en mode glissant
learning_rate = 0.8
discount_factor = 0.95

# Exploration
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.0005  # Décroissance lente pour bien explorer

# ==========================================
# 2. INITIALISATION
# ==========================================
print("--- Démarrage de l'entraînement (Mode Difficile / Glissant) ---")

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode=None)

# 16 états x 4 actions
q_table = np.zeros((env.observation_space.n, env.action_space.n)) # type: ignore

# Variable pour compter les victoires pendant l'entraînement
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

        # C. Bellman Update
        max_future_q = np.max(q_table[new_state, :])
        current_q = q_table[state, action]

        new_q = current_q + learning_rate * (
            float(reward) + discount_factor * max_future_q - current_q
        )
        q_table[state, action] = new_q

        # D. Compteur de victoire
        if reward == 1:
            success_count += 1

        state = new_state

    # E. Decay Epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    # Log tous les 1000 épisodes
    if (episode + 1) % 1000 == 0:
        print(
            f"Épisode {episode + 1} | Taux cumulé : {success_count / (episode + 1):.2%}"
        )

env.close()

# Résultat final de l'entraînement
success_rate = (success_count / total_episodes) * 100
print(f"\n--- Entraînement terminé ---")
print(f"Nombre de victoires (Training) : {success_count} / {total_episodes}")
print(f"Taux de réussite global (Training) : {success_rate:.2f}%")


# ==========================================
# 4. VISUALISATION DE LA Q-TABLE (HEATMAP)
# ==========================================
def plot_q_table(q_table):
    plt.figure(figsize=(10, 6))
    sns.heatmap(q_table, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title("Q-Table : Valeur de chaque action pour chaque état")
    plt.xlabel("Actions (0=Gauche, 1=Bas, 2=Droite, 3=Haut)")
    plt.ylabel("États (Cases du lac 0 à 15)")
    actions_labels = ["Gauche", "Bas", "Droite", "Haut"]
    plt.xticks(np.arange(4) + 0.5, actions_labels, rotation=0)
    plt.show()


print("\nAffichage de la Heatmap...")
plot_q_table(q_table)


# ==========================================
# 5. DÉMO FINALE (VISUELLE)
# ==========================================
print("\n--- Démarrage de la Démo Visuelle (5 essais) ---")
# On recrée l'environnement en mode HUMAN pour voir
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
        time.sleep(0.5)  # Ralenti pour l'oeil humain

env_visu.close()


# ==========================================
# 6. ÉVALUATION SCIENTIFIQUE (100 Épisodes)
# ==========================================
# C'est ici qu'on mesure la vraie performance sans le biais visuel ou d'apprentissage
print("\n--- Démarrage de l'Évaluation Statistique (100 Épisodes) ---")

# On utilise render_mode=None pour que ça aille vite
eval_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode=None)

nb_eval_episodes = 100
eval_wins = 0

for episode in range(nb_eval_episodes):
    state, _ = eval_env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # EXPLOITATION PURE : On prend la meilleure action connue
        action = np.argmax(q_table[state, :])
        state, reward, terminated, truncated, _ = eval_env.step(action)

        # On compte les points (sans toucher à la Q-Table !)
        if reward == 1:
            eval_wins += 1

eval_env.close()

# Score final
score_final = (eval_wins / nb_eval_episodes) * 100
print(
    f"Résultat final sur {nb_eval_episodes} tentatives : {score_final:.2f}% de réussite."
)
