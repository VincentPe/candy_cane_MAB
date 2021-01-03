
import numpy as np
import pandas as pd


# Initiate variables
initial_q_values = 0.5
step_size = 0.25
epsilon = 0.1

total_reward = 0
last_action = None
q_values = []
arm_counts = []


def argmax(q_values):
    """
    Takes in a list of q_values and returns the index of the item 
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []
    
    for i in range(len(q_values)):
        # if a value in q_values is greater than the highest value update top and reset ties to zero
        if q_values[i] > top_value:
            top_value = q_values[i]
            ties = [i]
        # if a value is equal to top value add the index to ties
        elif q_values[i] == top_value:
            ties.append(i)
    # return a random selection from ties. 
    return np.random.choice(ties)


def epsilon_greedy_agent(observation, configuration):
    """
    Takes one step for the agent. It takes in a reward and observation and 
    returns the action the agent chooses at that time step.
    """
    global q_values, arm_counts, last_action, total_reward
    
    #print(f'Received reward {observation.reward - total_reward} for action {last_action}.')
    
    if observation.step == 0:
        q_values = [initial_q_values] * configuration.banditCount
        arm_counts = [0] * configuration.banditCount
    else:
        # Get the latest reward
        reward = observation.reward - total_reward
        total_reward = observation.reward
        
        # Using a stepsize learning rate:
        q_values[last_action] += step_size * (reward - q_values[last_action])
        # Using a sample average: 
        # q_values[last_action] += 1/arm_counts[last_action] * (reward - q_values[last_action])
        
        q_values[last_action] = q_values[last_action] * configuration.decayRate
        
    if np.random.random() < epsilon:
        last_action = np.random.randint(len(q_values))
        #print(f'Taking action: {last_action} randomly.')
    else:
        last_action = int(argmax(q_values))
        #print(f'Taking action: {last_action}. Tried {arm_counts[last_action]} times before, '
        #      f'with q_value: {q_values[last_action]}.')
    
    arm_counts[last_action] += 1
    
    if observation.step == configuration.episodeSteps-2:
        print(f'Total reward earned: {total_reward}')
        arm_pulls = pd.DataFrame({'bandit': range(configuration.banditCount),
                                  'arm_pulls': arm_counts,
                                  'q_value': q_values})
        print(arm_pulls.sort_values(by='arm_pulls', ascending=False))
    
    return last_action
