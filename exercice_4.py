"""
Exercice 4: Entraînement d'un agent DQN avec Stable-Baselines3
Objectif: Entraîner, évaluer et sauvegarder un agent DQN sur CartPole-v1
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import os


def main():
    """Fonction principale pour entraîner et évaluer un agent DQN."""

    print("=" * 70)
    print("Entraînement d'un agent DQN sur CartPole-v1 avec Stable-Baselines3")
    print("=" * 70)

    # ========== 1. Création de l'environnement ==========
    print("\n[1] Création de l'environnement CartPole-v1...")
    env = gym.make("CartPole-v1")
    print(f"    ✓ Environnement créé")
    print(f"    Espace d'états: {env.observation_space}")
    print(f"    Espace d'actions: {env.action_space}")

    # ========== 2. Création des répertoires pour les logs ==========
    print("\n[2] Création des répertoires pour les logs...")
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)
    print("    ✓ Répertoires créés")

    # ========== 3. Instantiation du modèle DQN ==========
    print("\n[3] Instantiation du modèle DQN...")
    print("    Configuration:")
    print("    - Policy: MlpPolicy (réseau de neurones multicouche)")
    print("    - Verbose: 1 (affichage de la progression)")
    print("    - TensorBoard logs: ./logs/")

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )
    print("    ✓ Modèle DQN instancié")

    # ========== 4. Entraînement du modèle ==========
    print("\n[4] Entraînement du modèle DQN...")
    print("    Nombre de timesteps: 25000")
    print("    (Les timesteps sont des interactions avec l'environnement,")
    print("     pas des épisodes complets)")
    print()

    model.learn(total_timesteps=25000)
    print("\n    ✓ Entraînement terminé")

    # ========== 5. Évaluation du modèle ==========
    print("\n[5] Évaluation du modèle sur 100 épisodes...")
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=100, deterministic=True
    )
    print("    ✓ Évaluation terminée")
    print(f"    Récompense moyenne: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"    (Objectif: >195 pour CartPole-v1)")

    # ========== 6. Sauvegarde du modèle ==========
    print("\n[6] Sauvegarde du modèle...")
    model_path = "./models/dqn_cartpole"
    model.save(model_path)
    print(f"    ✓ Modèle sauvegardé: {model_path}.zip")

    # ========== 7. Affichage des résultats finaux ==========
    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"Récompense moyenne (100 épisodes): {mean_reward:.2f}")
    print(f"Écart-type: {std_reward:.2f}")
    print(f"Modèle sauvegardé: {os.path.abspath(model_path)}.zip")
    print("=" * 70)

    # ========== 8. Test sur quelques épisodes ==========
    print("\n[8] Test du modèle entraîné sur 5 épisodes...")
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            env.render()
            done = terminated or truncated

        print(f"    Episode {episode + 1}: Récompense = {episode_reward:.0f}")

    env.close()
    print("\n✓ Programme terminé avec succès!")


if __name__ == "__main__":
    main()
