import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RecyclingRobot:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1,
                 prob_down=0.2, prob_success=0.5,
                 r_wait=1, r_search=3):

        # Probabilidades do problema
        self.alpha = alpha        # taxa de aprendizado
        self.gamma = gamma        # fator de desconto
        self.epsilon = epsilon    # taxa de exploração
        self.prob_down = prob_down  # probabilidade de cair de high -> low após search
        self.prob_success = prob_success  # probabilidade de sucesso no low-search

        # Recompensas
        self.r_wait = r_wait
        self.r_search = r_search

        # Espaço de estados e ações
        self.states = ["high", "low"]
        self.actions = {
            "high": ["search", "wait"],
            "low": ["search", "wait", "recharge"]
        }

        # Inicializa Q-valores
        self.Q = {s: {a: 0 for a in self.actions[s]} for s in self.states}

        # Estado inicial
        self.state = "high"

    def reset(self):
        self.state = "high"

    def choose_action(self, state):
        """Política epsilon-greedy"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions[state])
        return max(self.Q[state], key=self.Q[state].get)

    def step(self, action):
        """Executa ação, retorna recompensa e próximo estado"""
        if self.state == "high":
            if action == "search":
                reward = self.r_search
                if np.random.rand() < self.prob_down:
                    next_state = "low"
                else:
                    next_state = "high"
            elif action == "wait":
                reward = self.r_wait
                next_state = "high"

        else:  # estado low
            if action == "search":
                if np.random.rand() < self.prob_success:
                    reward = self.r_search
                    next_state = "low"
                else:
                    reward = -3
                    next_state = "high"
            elif action == "wait":
                reward = self.r_wait
                next_state = "low"
            elif action == "recharge":
                reward = 0
                next_state = "high"

        return reward, next_state

    def temporal_diff(self, state, action, reward, next_state):
        """Atualização TD(0)"""
        best_next = max(self.Q[next_state], key=self.Q[next_state].get)
        td_target = reward + self.gamma * self.Q[next_state][best_next]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def train(self, epochs=50, steps=1000, save_file="rewards.txt"):
        rewards_per_epoch = []

        with open(save_file, "w") as f:
            for ep in range(epochs):
                self.reset()
                total_reward = 0
                for _ in range(steps):
                    state = self.state
                    action = self.choose_action(state)
                    reward, next_state = self.step(action)
                    self.temporal_diff(state, action, reward, next_state)
                    self.state = next_state
                    total_reward += reward

                rewards_per_epoch.append(total_reward)
                f.write(f"{total_reward}\n")

        return rewards_per_epoch


# ---------------------- Execução ----------------------

robot = RecyclingRobot(prob_down=0.2, prob_success=0.5,
                       r_wait=1, r_search=3)

rewards = robot.train(epochs=100, steps=1000)

# Plot curva de recompensa
plt.plot(rewards)
plt.xlabel("Épocas")
plt.ylabel("Recompensa total")
plt.title("Aprendizado do Robô de Reciclagem")
plt.show()

# Política ótima (heatmap)
policy = {s: max(robot.Q[s], key=robot.Q[s].get) for s in robot.states}
policy_matrix = [[robot.Q["high"]["search"], robot.Q["high"]["wait"]],
                 [robot.Q["low"]["search"], robot.Q["low"]["wait"]]]

sns.heatmap(policy_matrix, annot=True, fmt=".2f",
            xticklabels=["search", "wait"],
            yticklabels=["high", "low"], cmap="coolwarm")
plt.title("Q-valores aprendidos (Política ótima)")
plt.show()

