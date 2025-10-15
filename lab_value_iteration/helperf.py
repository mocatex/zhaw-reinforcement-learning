import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def make_env(is_slippery: bool = False, map_size: int = 5, proba_frozen: float = 0.9, seed: int = 0):
    """Create and return the environment."""
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=is_slippery,
        render_mode="rgb_array",
        desc=generate_random_map(
            size=map_size, p=proba_frozen, seed=seed
        ),
    )
    return env

def generate_greedy_episode(env, V, epsilon=0.1):
    s, _ = env.reset()
    traj = []
    done = False
    while not done:
        if np.random.rand() < epsilon:      # ε-greedy policy
            a = env.action_space.sample()    # random action
        else:
            q_values = []
            for a in range(env.action_space.n):
                s_next, r, term, trunc, _ = env.step(a)
                q_values.append(r + (0 if term or trunc else V[s_next]))
                env.env.s = s  # zurücksetzen
            a = np.argmax(q_values)           # beste Aktion wählen
        s_next, r, term, trunc, _ = env.step(a)
        traj.append((s, a, r))
        s = s_next
        done = term or trunc
    return traj

def generate_random_episode(env):
    s, _ = env.reset()
    traj = []
    done = False
    while not done:
        a = env.action_space.sample()          # random policy
        s_next, r, term, trunc, _ = env.step(a)
        traj.append((s, a, r))
        s = s_next
        done = term or trunc
    return traj

def monte_carlo_V(env, episodes:int=10000, gamma:float=0.1, first_visit:bool=True, incremental:bool=True):
    V = defaultdict(float)
    N = defaultdict(int)

    for episode in range(episodes):
        traj = generate_random_episode(env) if incremental else generate_greedy_episode(env, V, gamma)
        states = [s for s, _, _ in traj]
        rewards = [r for _, _, r in traj]

        G = 0.0
        seen = set()
        # rückwärts: akkumuliert Return; update nur beim ersten Besuch (from start)
        for t in range(len(traj)-1, -1, -1):
            G = gamma * G + rewards[t]
            s = states[t]
            if first_visit and (s in seen):  # Wenn State schon besucht, ignorieren
                continue
            seen.add(s)
            N[s] += 1
            V[s] += (G - V[s]) / N[s]          # inkrementeller Mittelwert

        if episode % 100 == 0:
            print(f"Episode: {episode}")
            print(f"Reward: {sum(rewards)}")
            print(f"Lenght Trajectory: {len(traj)}")
    env.close()
    return V, N

def print_heat_map(env, V, N):
    desc = env.unwrapped.desc  # Bytes-Array
    H, W = desc.shape

    # Value-Grid füllen
    V_grid = np.full((H, W), np.nan, dtype=float)
    N_grid = np.zeros((H, W), dtype=int)
    for s, v in V.items():
        r, c = divmod(s, W)
        V_grid[r, c] = v
    for s, n in N.items():
        r, c = divmod(s, W)
        N_grid[r, c] = n

     # Maske für Löcher und Ziel
    mask_hole = np.zeros((H, W), dtype=bool)
    mask_goal = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if desc[r, c] == b'H':
                mask_hole[r, c] = True
            if desc[r, c] == b'G':
                mask_goal[r, c] = True

    # Eigene Farbkarte: Werte, Loch (blau), Ziel (gold)
    cmap = plt.cm.viridis
    cmap = cmap.copy()
    cmap.set_bad(color='blue')   # Löcher
    cmap.set_over(color='gold')  # Ziel

    # Löcher als NaN, Ziel als sehr hoher Wert
    V_grid[mask_hole] = np.nan
    V_grid[mask_goal] = np.nanmax(V_grid) + 1

    plt.figure()
    im = plt.imshow(V_grid, origin="upper", cmap=cmap, vmin=np.nanmin(V_grid), vmax=np.nanmax(V_grid))
    plt.colorbar(im, label="Erwarteter Return V(s)")
    plt.xticks(range(W)); plt.yticks(range(H))
    plt.gca().set_aspect("equal")

    for r in range(H):
        for c in range(W):
            if mask_hole[r, c]:
                plt.text(c, r, "H", ha="center", va="center", fontsize=10, color="white", weight="bold")
            elif mask_goal[r, c]:
                plt.text(c, r, "G", ha="center", va="center", fontsize=10, color="black", weight="bold")
            elif not np.isnan(V_grid[r, c]):
                plt.text(c, r, f"{V_grid[r,c]:.2f}\nN={N_grid[r,c]}", ha="center", va="center", fontsize=8)

    plt.title("FrozenLake Value-Heatmap (First-Visit MC, random policy)")
    plt.tight_layout()
    plt.show()
