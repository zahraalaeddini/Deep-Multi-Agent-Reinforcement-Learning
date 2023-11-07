import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Define the factors
factors_1 = ['E2F1', 'NFKB1', 'SRF', 'RELA', 'SP1', 'AR', 'HDAC2', 'EGR1', 'PAX5', 'MYB']
factors_2 = ['155', '124-3', '124', '16-2', '1343', '146a', '16-1', '7b', '17', '27a', '34a', '1-2', '107', '1-1']

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Deep Q-Network (DQN) agent
class DQNAgent:
    def __init__(self, num_inputs, num_actions, alpha, gamma, epsilon):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = QNetwork(num_inputs, num_actions)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if np.random.uniform() < self.epsilon:
            # Explore: randomly select an action
            action = np.random.randint(self.num_actions)
        else:
            # Exploit: select the action with the maximum Q-value
            with torch.no_grad():
                q_values = self.q_network(torch.Tensor(state))
                action = torch.argmax(q_values).item()
        return action

    def update_q_network(self, state, action, reward, next_state, done):
        self.optimizer.zero_grad()
        q_values = self.q_network(torch.Tensor(state))
        next_q_values = self.q_network(torch.Tensor(next_state))
        q_value = q_values[action]
        if done:
            target_q_value = torch.Tensor([reward])
        else:
            target_q_value = torch.Tensor([reward + self.gamma * torch.max(next_q_values).item()])
        loss = self.criterion(q_value, target_q_value)
        loss.backward()
        self.optimizer.step()

# Define the agent's initial positions
agent_positions = [(0, 0), (len(factors_1) - 1, len(factors_2) - 1)]

# Define the play parameters
num_plays = 10
max_steps = 100

# Define the hyperparameters
alpha = 0.001  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Track the positions of the agents after each play
play_positions = []

# Play loop
for play in range(num_plays):
    print(f"Play {play + 1}/{num_plays}")

    # Reset the agent positions
    agent_positions = [(0, 0), (len(factors_1) - 1, len(factors_2) - 1)]

    # Create DQN agents
    num_agents = len(agent_positions)
    agents = [DQNAgent(2, 4, alpha, gamma, epsilon) for _ in range(num_agents)]

    for step in range(max_steps):
        # Select an action for each agent
        agent_actions = []
        for agent_id in range(num_agents):
            position = agent_positions[agent_id]
            x, y = position
            state = [x, y]
            action = agents[agent_id].select_action(state)
            agent_actions.append(action)

            # Move the agent
            if action == 0:  # Activate
                if x > 0:
                    x -= 1
            elif action == 1:  # Inhibit
                if x < len(factors_1) - 1:
                    x += 1
            elif action == 2:  # Bind
                if y > 0:
                    y -= 1
            elif action == 3:  # Inactivate
                if y < len(factors_2) - 1:
                    y += 1

            # Update the agent's position
            agent_positions[agent_id] = (x, y)

        # Update the Q-values
        new_positions = agent_positions.copy()
        for agent_id in range(num_agents):
            x, y = new_positions[agent_id]
            reward = 0  # Define the reward based on the factors and other conditions

            # Update the Q-network
            agents[agent_id].update_q_network([x, y], agent_actions[agent_id], reward, [x, y], False)

            # Check if the play is finished
            if (x, y) == (len(factors_1) - 1, len(factors_2) - 1):
                break

        if (x, y) == (len(factors_1) - 1, len(factors_2) - 1):
            print("Play finished!")
            break

    # Add the final positions of the agents to the list
    play_positions.append(agent_positions.copy())

print("Training complete!")

# Track the last four final actions for each agent
last_four_actions = []

# Play loop
for play in range(num_plays):
    # ...

    # Reset the agent positions
    agent_positions = [(0, 0), (len(factors_1) - 1, len(factors_2) - 1)]

    for step in range(max_steps):
        # ...

        # Update the Q-values
        new_positions = agent_positions.copy()
        for agent_id in range(num_agents):
            # ...

            # Check if the play is finished
            if (x, y) == (len(factors_1) - 1, len(factors_2) - 1):
                break

        if (x, y) == (len(factors_1) - 1, len(factors_2) - 1):
            print("Play finished!")
            last_four_actions.append(agent_actions[-4:])  # Store the last four actions for each agent
            break

# Print the last four final actions for each agent
for agent_id, actions in enumerate(last_four_actions):
    print(f"Agent {agent_id + 1}: {actions}")


# Plot the positions of the agents after each play
for play, positions in enumerate(play_positions):
    plt.figure()
    for agent_id, position in enumerate(positions):
        x, y = position
        plt.plot(y, x, 'o', label=f'Agent {agent_id + 1}')
    plt.xticks(range(len(factors_2)), factors_2, rotation = 90)
    plt.yticks(range(len(factors_1)), factors_1)
    plt.title(f'Play {play + 1}')
    plt.legend()
    plt.grid(True)
    plt.show()


