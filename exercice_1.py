from typing import SupportsFloat
import gymnasium as gym

# --- 1. Création de l'instance (Lancement du jeu) ---
env = gym.make("CartPole-v1", render_mode="human")

# --- 2. Inspection de l'INPUT (Ce que l'agent voit) ---
print("=== ESPACE D'OBSERVATION (INPUT) ===")
# On récupère l'objet 'Space'
obs_space = env.observation_space

print(f"Type : {obs_space}")
# Attendu : Box(4) -> Un tableau de 4 chiffres continus

print(f"Bornes Min (Low) : \n{obs_space.low}")  # type: ignore
# Affiche les valeurs minimales théoriques pour [Pos, Vit, Angle, VitAng]

print(f"Bornes Max (High) : \n{obs_space.high}")  # type: ignore
# Affiche les valeurs maximales théoriques

# On tire une observation au hasard pour voir à quoi ça ressemble
print(f"Exemple aléatoire (Sample) : \n{obs_space.sample()}")


# --- 3. Inspection de l'OUTPUT (Ce que l'agent fait) ---
print("\n=== ESPACE D'ACTION (OUTPUT) ===")
# On récupère l'objet 'Space'
action_space = env.action_space

print(f"Type : {action_space}")
# Attendu : Discrete(2) -> Un choix binaire

print(f"Nombre d'actions : {action_space.n}")  # type: ignore
# Attendu : 2 (0 ou 1)

# On tire une action au hasard
print(f"Exemple d'action aléatoire : {action_space.sample()}")


print("--- Début de la simulation (10 Épisodes) ---")

# BOUCLE EXTERNE : Gère les épisodes (les parties complètes)
for episode in range(1, 11):

    # A. Reset : On remet le jeu à zéro pour commencer
    # Note : reset() renvoie un tuple (observation, info), on ignore 'info' ici avec _
    observation, _ = env.reset()

    total_reward: float = 0.0
    terminated: bool = False  # Le jeu est fini (gagné ou perdu)
    truncated: bool = False  # Le temps est écoulé (limite de steps atteinte)

    # BOUCLE INTERNE : La "Game Loop" (tant que la partie n'est pas finie)
    while not terminated and not truncated:

        # B. Politique : On choisit une action au hasard (0 ou 1)
        action = env.action_space.sample()

        # C. Step : On applique l'action dans l'environnement
        # On récupère 5 variables cruciales
        new_obs, reward, terminated, truncated, info = env.step(action)

        # D. Accumulation du score
        total_reward += float(reward)

        # (Optionnel) Mise à jour de l'observation pour le tour suivant
        observation = new_obs

    # Fin de l'épisode
    print(f"Épisode {episode} : Récompense Totale = {total_reward}")

# Fermeture propre
env.close()
print("--- Fin de la simulation ---")
