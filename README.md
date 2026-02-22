# Projet d'Introduction à l'Apprentissage par Renforcement

Ce projet contient quatre exercices d'introduction à l'apprentissage par renforcement avec les bibliothèques Gymnasium, PyTorch et Stable-Baselines3.

## Packages Utilisés

Ce projet utilise les packages Python suivants :

- **gymnasium[classic-control]**: La bibliothèque de base pour les environnements d'apprentissage par renforcement.
- **matplotlib**: Pour la visualisation des données, notamment la Q-table.
- **numpy**: Pour les calculs numériques et la gestion des tableaux.
- **seaborn**: Pour créer des visualisations statistiques plus esthétiques (heatmap).
- **stable-baselines3**: Une bibliothèque populaire contenant des implémentations d'algorithmes d'apprentissage par renforcement.
- **torch**: Framework PyTorch pour construire des réseaux de neurones profonds.
- **tensorboard**: Pour la visualisation des métriques d'entraînement.

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
3.  **Visualisation** : Après l'entraînement, une _heatmap_ de la Q-table est générée avec `matplotlib` et `seaborn` pour visualiser la politique apprise par l'agent.
4.  **Démonstration** : Le script lance 5 épisodes en mode visuel pour montrer l'agent utiliser sa politique apprise pour naviguer sur le lac.
5.  **Évaluation** : Enfin, une évaluation statistique est menée sur 100 épisodes (sans exploration) pour mesurer la performance réelle de l'agent. Le taux de réussite final est ensuite affiché.

### `exercice_3.py`

Ce script implémente les deux classes fondamentales pour le **Deep Q-Learning (DQN)** avec PyTorch.

**Objectif :** Créer une implémentation manuelle des composants nécessaires au DQN.

#### Classe `DQN` (héritant de `torch.nn.Module`)

Le réseau de neurones pour Deep Q-Learning qui approxime les Q-valeurs.

**Structure du réseau :**

- Couche d'entrée : `nn.Linear(state_size, hidden_size)`
- Couche cachée 1 : `nn.Linear(hidden_size, hidden_size)` avec activation ReLU
- Couche cachée 2 : `nn.Linear(hidden_size, hidden_size)` avec activation ReLU
- Couche de sortie : `nn.Linear(hidden_size, action_size)` (sans activation)

**Méthode principale :**

- `forward(state)` : Propage l'état à travers les couches avec activation ReLU entre elles et retourne les Q-valeurs pour chaque action.

#### Classe `ReplayBuffer`

Buffer de mémoire de rejeu pour stocker et échantillonner les transitions d'apprentissage.

**Fonctionnalités :**

- **`__init__(capacity)`** : Initialise un `collections.deque` avec une capacité fixe (par défaut 10 000 transitions).
- **`push(state, action, reward, next_state, done)`** : Ajoute une expérience (tuple) au buffer.
- **`sample(batch_size)`** : Extrait un batch aléatoire du buffer en utilisant `random.sample()`.
- **`__len__()`** : Retourne le nombre de transitions actuellement stockées.

**Utilisation :** Ces classes forment la base pour construire un agent DQN personnalisé.

**Exemple d'exécution fourni :** Le script contient une démonstration complète avec création du réseau, ajout de transitions au buffer et extraction de samples.

### `exercice_4.py`

Ce script entraîne, évalue et sauvegarde un agent **DQN** de Stable-Baselines3 sur l'environnement **CartPole-v1**.

**Objectif :** Démontrer comment utiliser les algorithmes d'apprentissage par renforcement modernes de la bibliothèque Stable-Baselines3.

**Étapes du script :**

1. **Création de l'environnement** : Initialisation de CartPole-v1 avec Gymnasium.

2. **Instantiation du modèle DQN** :
    - Politique : `MlpPolicy` (réseau multicouche)
    - Apprentissage avec logs TensorBoard pour le suivi des métriques
    - Configuration : learning_rate=1e-3, buffer_size=10000, batch_size=32

3. **Entraînement** : Le modèle est entraîné sur **25 000 timesteps** (interactions avec l'environnement, pas d'épisodes).

4. **Évaluation** : Le modèle entraîné est évalué sur **100 épisodes** pour mesurer sa performance moyenne.

5. **Sauvegarde** : Le modèle est sauvegardé au format `.zip` dans le répertoire `./models/`.

6. **Visualisation des logs** : Les métriques d'entraînement sont enregistrées dans `./logs/` et peuvent être visualisées avec TensorBoard.

**Résultats attendus :**

- Modèle entraîné et sauvegardé
- Récompense moyenne évaluée sur 100 épisodes
- Logs d'entraînement avec TensorBoard (loss, récompenses, longueur d'épisode, etc.)

**Point important :** Ne pas confondre le DQN manuel (exercice 3) avec le DQN de Stable-Baselines3 (exercice 4). Le premier illustre les concepts, le second utilise une implémentation optimisée et prête à l'emploi.

## Visualisation des Résultats

### TensorBoard (Exercice 4)

Pour visualiser les métriques d'entraînement du DQN :

```bash
source .venv/bin/activate
tensorboard --logdir=./logs/
```

Puis ouvrez `http://localhost:6006` dans votre navigateur pour voir :

- **Graphiques de perte** (loss)
- **Récompenses moyennes** par épisode
- **Longueur des épisodes**
- **Taux d'exploration**
- **FPS et temps d'entraînement**

## Installation et Utilisation

### Premier démarrage

```bash
# Installation des dépendances
uv sync

# Activation de l'environnement virtuel
source .venv/bin/activate
```

### Exécution des exercices

```bash
# Exercice 1 : Exploration basique de CartPole
python exercice_1.py

# Exercice 2 : Q-learning sur FrozenLake
python exercice_2.py

# Exercice 3 : Implémentation manuelle DQN + ReplayBuffer
python exercice_3.py

# Exercice 4 : Entraînement DQN de Stable-Baselines3
python exercice_4.py
```

### Visualisation TensorBoard

```bash
source .venv/bin/activate
tensorboard --logdir=./logs/
# Ouvrez http://localhost:6006 dans votre navigateur
```

## Fichiers Générés

Après l'exécution de l'exercice 4 :

- **`./models/dqn_cartpole.zip`** : Modèle DQN entraîné et sauvegardé
- **`./logs/DQN_1/`** : Événements TensorBoard pour visualisation des métriques d'entraînement

## Progression Pédagogique

1. **Exercice 1** : Comprendre les bases (état, action, récompense)
2. **Exercice 2** : Implémenter un algorithme classique (Q-learning)
3. **Exercice 3** : Construire les composants du Deep Learning (DQN + ReplayBuffer)
4. **Exercice 4** : Utiliser des implémentations optimisées (Stable-Baselines3)

Chaque exercice s'appuie sur les concepts précédents pour progresser vers des algorithmes plus avancés.
