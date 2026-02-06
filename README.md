# Projet d'Introduction à l'Apprentissage par Renforcement

Ce projet contient deux exercices d'introduction à l'apprentissage par renforcement avec la bibliothèque Gymnasium.

## Packages Utilisés

Ce projet utilise les packages Python suivants :

- **gymnasium[classic-control]**: La bibliothèque de base pour les environnements d'apprentissage par renforcement.
- **matplotlib**: Pour la visualisation des données, notamment la Q-table.
- **numpy**: Pour les calculs numériques et la gestion des tableaux.
- **seaborn**: Pour créer des visualisations statistiques plus esthétiques (heatmap).
- **stable-baselines3**: Une bibliothèque populaire contenant des implémentations d'algorithmes d'apprentissage par renforcement (bien que non utilisée directement dans les exercices, elle est listée dans les dépendances).

## Scripts

### `exercice_1.py`

Ce script est une première exploration de l'environnement **CartPole-v1**.

**Objectif :** Se familiariser avec les concepts de base de Gymnasium.

Le script effectue les actions suivantes :
1.  Il initialise l'environnement `CartPole-v1`.
2.  Il inspecte et affiche les caractéristiques de l'espace d'observation (ce que l'agent "voit") et de l'espace d'action (ce que l'agent "peut faire").
3.  Il lance une simulation de 10 épisodes où l'agent choisit des actions de manière complètement aléatoire.
4.  Il affiche la récompense totale obtenue à la fin de chaque épisode.

### `exercice_2.py`

Ce script implémente un algorithme de **Q-learning** pour résoudre l'environnement **FrozenLake-v1**.

**Objectif :** Entraîner un agent à trouver le chemin du point de départ (S) à l'objectif (G) sur un lac gelé, en évitant les trous (H).

Le script se déroule en plusieurs étapes :
1.  **Configuration** : Définition des hyperparamètres comme le taux d'apprentissage, le facteur de dépréciation et la stratégie d'exploration (epsilon-greedy).
2.  **Entraînement** : L'agent est entraîné sur 15 000 épisodes. Durant chaque épisode, il explore l'environnement et met à jour une **Q-table** en utilisant l'équation de Bellman pour apprendre la valeur de chaque action dans chaque état.
3.  **Visualisation** : Après l'entraînement, une *heatmap* de la Q-table est générée avec `matplotlib` et `seaborn` pour visualiser la politique apprise par l'agent.
4.  **Démonstration** : Le script lance 5 épisodes en mode visuel pour montrer l'agent utiliser sa politique apprise pour naviguer sur le lac.
5.  **Évaluation** : Enfin, une évaluation statistique est menée sur 100 épisodes (sans exploration) pour mesurer la performance réelle de l'agent. Le taux de réussite final est ensuite affiché.
