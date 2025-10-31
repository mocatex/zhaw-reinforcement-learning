import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium import Wrapper
import numpy as np
from collections import defaultdict

class RewardShapingWrapper(Wrapper):
    """
    Wrapper für FrozenLake:
    - Agent fällt in Loch: -10
    - normaler Schritt: -0.01
    - läuft in Wand (kein Positionswechsel): -1
    - erreicht Goal: +10
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # aktuellen Basiszustand sichern (falls vorhanden)
        prev_s = getattr(self.env.unwrapped, "s", None)

        s_next, _, terminated, truncated, info = self.env.step(action)

        # Beschreibung und Spaltenanzahl vom Basis-Env
        desc = self.env.unwrapped.desc
        ncol = self.env.unwrapped.ncol
        r_new = -0.01  # default Schrittstrafe

        # Zustandskoordinaten des neuen Zustands
        row, col = divmod(int(s_next), ncol)
        cell = desc[row, col]

        if cell == b'H':
            r_new = -10.0
        elif cell == b'G':
            r_new = 10.0
        else:
            # Wand: kein Positionswechsel gegenüber prev_s
            if prev_s is not None and int(s_next) == int(prev_s):
                r_new = -1.0

        return int(s_next), r_new, terminated, truncated, info

def make_env(is_slippery: bool = False, map_size: int = 5, proba_frozen: float = 0.9, seed: int = 0):
    """Hilfsfunktion: erzeugt FrozenLake und wickelt ihn mit RewardShapingWrapper."""
    base_env = gym.make(
        "FrozenLake-v1",
        is_slippery=is_slippery,
        render_mode=None,
        desc=generate_random_map(size=map_size, p=proba_frozen, seed=seed),
    )
    return RewardShapingWrapper(base_env)

def generate_random_episode(env, max_steps: int = 1000, render: bool = False):
    """
    Erzeuge eine zufällige Episode.
    - env: Gym-Umgebung (ggf. mit Wrapper)
    - max_steps: abs. Obergrenze, um Endlosschleifen zu vermeiden
    - render: wenn True, wird nach jedem Schritt versucht zu rendern
    Rückgabe: Liste von Tuples (state:int, action:int, reward:float)
    """
    s, _ = env.reset()
    traj = []
    done = False
    steps = 0

    while not done and steps < max_steps:
        a = env.action_space.sample()
        s_next, r, term, trunc, _ = env.step(a)

        traj.append((int(s), int(a), float(r)))

        s = s_next
        done = term or trunc
        steps += 1

        if render:
            try:
                env.render()
            except Exception:
                pass

    return traj

def shaped_reward(prev_s, s_next, desc, ncol):
    row, col = divmod(int(s_next), ncol)
    cell = desc[row, col]
    if cell == b'H':  # Loch
        return -10.0
    if cell == b'G':  # Goal
        return 10.0
    # Wand?
    return -1.0 if int(s_next) == int(prev_s) else -0.01

def generate_greedy_episode(env, V, epsilon=0.1, gamma=0.9, max_steps=1000):
    s, _ = env.reset()
    traj, done, steps = [], False, 0

    P = env.unwrapped.P
    desc = env.unwrapped.desc
    ncol = env.unwrapped.ncol

    while not done and steps < max_steps:
        if np.random.rand() < epsilon:
            a = int(env.action_space.sample())
        else:
            q_vals = []
            for a_cand in range(env.action_space.n):
                q = 0.0
                for prob, s_next, _, done_flag in P[int(s)][a_cand]:
                    r = shaped_reward(int(s), int(s_next), desc, ncol)
                    v_next = 0.0 if done_flag else gamma * V.get(int(s_next), 0.0)
                    q += prob * (r + v_next)
                q_vals.append(q)
            # tie-break verhindern
            a = int(np.argmax(q_vals) if len(set(q_vals))>1 else np.random.randint(env.action_space.n))

        s_next, r, term, trunc, _ = env.step(a)  # echter Schritt
        traj.append((int(s), int(a), float(r)))
        s = s_next
        done = bool(term or trunc)
        steps += 1
    return traj


def run_mc_experiment(
    env,
    episodes: int = 1000,
    alpha: float = 0.0,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    epsilon_decay: float = 0.995,
    first_visit: bool = True,
    greedy: bool = False,
    max_steps: int = 1000,
):
    """
    Monte‑Carlo Experiment:
    - env: Gym FrozenLake (ggf. mit Wrapper)
    - episodes: Anzahl Episoden
    - alpha: konstante Lernrate (wenn 0 -> sample average wird benutzt)
    - gamma: Discount
    - epsilon: Startepsilon für epsilon-greedy
    - epsilon_decay: Faktor, mit dem epsilon nach jeder Episode multipliziert wird
    - first_visit: wenn True, nur erstes Auftreten eines Zustands in Episode updaten
    - greedy: wenn True, benutzt generate_greedy_episode_fixed, sonst generate_random_episode
    - max_steps: Sicherheitslimit pro Episode (weitergeleitet an greedy/random falls nötig)
    Rückgabe: (V, N)
    """
    V = defaultdict(float)
    N = defaultdict(int)

    # Wähle Episode-Generator
    episode_generator = generate_greedy_episode if greedy else generate_random_episode

    episode_returns = []
    episode_lengths = []

    for ep in range(episodes):
        # Erzeuge Episode
        if greedy:
            traj = episode_generator(env, V, epsilon=epsilon, max_steps=max_steps)
        else:
            traj = episode_generator(env)

        states = [s for s, _, _ in traj]
        rewards = [r for _, _, r in traj]

        G = 0.0
        seen = set()
        # Rückwärts Durchlauf zum Berechnen von Returns
        for t in range(len(traj) - 1, -1, -1):
            G = gamma * G + rewards[t]
            s = states[t]
            if first_visit and (s in seen):
                continue
            seen.add(s)

            # Zähle Besuch für Statistik
            N[s] += 1

            # Update: konstante Lernrate alpha oder sample-average
            if alpha and alpha > 0.0:
                V[s] += alpha * (G - V[s])
            else:
                # inkrementeller Mittelwert
                V[s] += (G - V[s]) / N[s]

        # Statistik sammeln
        episode_returns.append(sum(rewards))
        episode_lengths.append(len(traj))

        # Epsilon decays nach Episode
        epsilon *= epsilon_decay

        # Optionale Fortschrittsausgabe kurz (alle 100 Episoden)
        if (ep + 1) % 100 == 0 or (ep + 1) == episodes:
            avg_ret = float(np.mean(episode_returns[-100:])) if len(episode_returns) >= 1 else 0.0
            avg_len = float(np.mean(episode_lengths[-100:])) if len(episode_lengths) >= 1 else 0.0
            print(f"Episode {ep+1}/{episodes} — letzte Return: {episode_returns[-1]:.3f}, Länge: {episode_lengths[-1]} — avg(last100) return: {avg_ret:.3f}, len: {avg_len:.2f}")

    env.close()
    return V, N, episode_returns, episode_lengths

# python
def plot_experiment_results(env, V, N, episode_returns, episode_lengths, gamma: float = 0.9, figsize=(12, 10)):
    """
    Zeichnet 4 Plots in einem Figure:
    1) Verlauf der Episodenlängen
    2) Verlauf der Returns
    3) Heatmap: V(s) mit Annotation N(s)
    4) Policy-Map: beste Aktion pro Tile als Pfeil (←, ↓, →, ↑)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    desc = env.unwrapped.desc
    H, W = desc.shape

    # Raster für V und N
    V_grid = np.full((H, W), np.nan, dtype=float)
    N_grid = np.zeros((H, W), dtype=int)
    for s, v in V.items():
        r, c = divmod(int(s), W)
        V_grid[r, c] = float(v)
    for s, n in N.items():
        r, c = divmod(int(s), W)
        N_grid[r, c] = int(n)

    mask_hole = desc == b'H'
    mask_goal = desc == b'G'

    # Goal sichtbar machen (höherer Wert), Löcher als NaN
    V_grid[mask_hole] = np.nan
    if np.any(~np.isnan(V_grid)):
        V_grid[mask_goal] = np.nanmax(V_grid) + 1.0
    else:
        V_grid[mask_goal] = 1.0

    # Prepare figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # 1) Episodenlängenverlauf
    ax1.plot(np.arange(1, len(episode_lengths) + 1), episode_lengths, color="tab:blue", alpha=0.8)
    ax1.set_title("Episode Länge")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Länge")

    # 2) Returns Verlauf
    ax2.plot(np.arange(1, len(episode_returns) + 1), episode_returns, color="tab:green", alpha=0.8)
    ax2.set_title("Episode Return")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Return")

    # 3) Heatmap V(s) mit N Annotation
    # sichere vmin/vmax
    finite = V_grid[~np.isnan(V_grid)]
    if finite.size > 0:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    else:
        vmin, vmax = 0.0, 1.0

    im = ax3.imshow(V_grid, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="V(s)")
    ax3.set_title("Value Heatmap (annotiert mit V / N)")
    ax3.set_xticks(np.arange(W)); ax3.set_yticks(np.arange(H))
    ax3.set_aspect("equal")

    for r in range(H):
        for c in range(W):
            if mask_hole[r, c]:
                ax3.text(c, r, "H", ha="center", va="center", color="white", weight="bold")
            elif mask_goal[r, c]:
                ax3.text(c, r, "G", ha="center", va="center", color="black", weight="bold")
            else:
                v = V_grid[r, c]
                n = N_grid[r, c]
                if np.isnan(v):
                    txt = f"N={n}"
                else:
                    txt = f"{v:.2f}\nN={n}"
                ax3.text(c, r, txt, ha="center", va="center", fontsize=8, color="white" if not np.isnan(v) and v < (vmin+vmax)/2 else "black")

    # 4) Policy-Map (greedy aus V) mit Pfeilen
    action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    policy_grid = np.full((H, W), "", dtype=object)

    # sicheren Originalzustand wiederherstellen nach Simulation
    orig_state = getattr(env.unwrapped, "s", None)
    orig_last = getattr(env.unwrapped, "lastaction", None)

    for s in range(H * W):
        r, c = divmod(s, W)
        if mask_hole[r, c]:
            policy_grid[r, c] = "H"
            continue
        if mask_goal[r, c]:
            policy_grid[r, c] = "G"
            continue

        q_values = []
        for a in range(env.action_space.n):
            # temporär setzen, Aktion simulieren
            env.unwrapped.s = int(s)
            env.unwrapped.lastaction = None
            s_next, reward, term, trunc, _ = env.step(a)
            future = 0.0 if (term or trunc) else gamma * float(V.get(int(s_next), 0.0))
            q_values.append(float(reward) + future)

        best_a = int(np.argmax(q_values))
        policy_grid[r, c] = action_symbols.get(best_a, "?")

    # restore
    if orig_state is not None:
        env.unwrapped.s = orig_state
    if orig_last is not None:
        env.unwrapped.lastaction = orig_last

    # Darstellung der Policy
    canvas = np.zeros((H, W))
    ax4.imshow(canvas, cmap="Greys", origin="upper")
    ax4.set_title("Greedy Policy aus V")
    ax4.set_xticks(np.arange(W)); ax4.set_yticks(np.arange(H))
    ax4.set_aspect("equal")
    for r in range(H):
        for c in range(W):
            ax4.text(c, r, policy_grid[r, c], ha="center", va="center", fontsize=18, weight="bold")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    """
    env_a_random_small = make_env(is_slippery=False, map_size=5, proba_frozen=0.85, seed=42)
    V_a, N_a, returns_a, lengths_a = run_mc_experiment(
        env_a_random_small,
        episodes=10000,
        alpha=0.1,
        gamma=0.9,
        first_visit=True,
        greedy=False,
        max_steps=200,
    )
    plot_experiment_results(env_a_random_small, V_a, N_a, returns_a, lengths_a, gamma=0.9)

    env_a_random_big = make_env(is_slippery=False, map_size=11, proba_frozen=0.85, seed=42)
    V_b, N_b, returns_b, lengths_b = run_mc_experiment(
        env_a_random_big,
        episodes=30000,
        alpha=0.1,
        gamma=0.9,
        first_visit=True,
        greedy=False,
        max_steps=200,
    )
    plot_experiment_results(env_a_random_big, V_b, N_b, returns_b, lengths_b, gamma=0.9)

    env_b_greedy_small = make_env(is_slippery=False, map_size=5, proba_frozen=0.85, seed=42)
    V, N, returns, lengths = run_mc_experiment(
        env_b_greedy_small,
        episodes=10000,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.9,
        epsilon_decay=0.99,
        first_visit=True,
        greedy=True,
        max_steps=200,
    )
    plot_experiment_results(env_b_greedy_small, V, N, returns, lengths, gamma=0.9)
    """
    env_b_greedy_big = make_env(is_slippery=False, map_size=11, proba_frozen=0.85, seed=42)
    V, N, returns, lengths = run_mc_experiment(
        env_b_greedy_big,
        episodes=30000,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.999,
        first_visit=True,
        greedy=True,
        max_steps=200,
    )
    plot_experiment_results(env_b_greedy_big, V, N, returns, lengths, gamma=0.9)