# Reinforcement Learning 
## Projeto 1

#### Professor: Flávio Codeço
#### Integrantes: Bryan Monteiro, Nicholas Costa e Sofia Monteiro


### Relatório
#### 1. Introdução

O projeto implementa um robô de reciclagem que utiliza Q-Learning para aprender a tomar decisões ótimas em diferentes estados de energia, maximizando suas recompensas.

O robô opera em dois estados de energia: high, alta energia, e low, baixa energia.

#### 2. Ações Disponíveis

- High
  - `search`: Procurar lixo, com recompensa 3 e 20% chance de ir para Low
  - `wait`: Aguardar, com recompensa 1 e permanece High

- Low
  - `search`: Procurar lixo, com 50% sucesso e 50% falha; em caso de sucesso, recompensa 3, já em falha, recompensa -3. Vai para High.
  - `wait`: Aguardar, com recompensa 1, permanece Low.
  - `recharge`: Recarregar, com recompensa 0, vai para High

#### 3. Implementação

- Algoritmo Q-Learning

  Utiliza a atualização TD(0):

  $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \cdot max Q(s',a') - Q(s,a)]$

- Parâmetros
  - Taxa de aprendizado $\alpha: 0.05$
  - Fator de desconto $\gamma: 0.9$
  - Taxa de exploração $\epsilon: 0.1$

 - Funções Principais

    - `choose_action()`: Política epsilon-greedy para seleção de ações
    - `step()`: Executa ação e retorna recompensa/próximo estado
    - `temporal_diff()`: Atualiza valores Q
    - `train()`: Loop principal de treinamento

#### 4. Configuração

- Treinamento: 1000 épocas, 1000 passos por época
- Inicialização: Valores Q = 20000
- Saída: Arquivo `rewards.txt` com recompensas por época

#### 5. Visualização

- Gráfico da recompensa total por época com curva de aprendizado
- Heatmap das probabilidades de ação por estado
