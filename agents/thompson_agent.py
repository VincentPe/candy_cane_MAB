
import numpy as np
import pandas as pd


# Initiate variables
initial_alpha = 1.  # total reward from bandit
initial_beta = 1.  # total losses from bandit

total_reward = 0
last_action = None
arm_counts = []
alphas = []
betas = []


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


def thompson_agent(observation, configuration):
    """
    Takes one step for the agent. It takes in a reward and observation and 
    returns the action the agent chooses at that time step based on bayesian sample.
    """
    global total_reward, last_action, arm_counts, alphas, betas
    
    if observation.step == 0:
        alphas = [initial_alpha] * configuration.banditCount
        betas = [initial_beta] * configuration.banditCount
        arm_counts = [0] * configuration.banditCount
    else:
        reward = observation.reward - total_reward  # Get the latest reward
        total_reward = observation.reward
        
        if reward:
            alphas[last_action] += 1
        else:
            betas[last_action] += 1
        
        # add decay rate
        alphas[last_action] = alphas[last_action] * configuration.decayRate
    
    probas = np.random.beta(alphas, betas)
    last_action = int(argmax(probas))
    arm_counts[last_action] += 1
    
    if observation.step == configuration.episodeSteps-2:
        print(f'Total reward earned: {total_reward}')
        arm_pulls = pd.DataFrame({'bandit': range(configuration.banditCount),
                                  'arm_pulls': arm_counts,
                                  'alpha': alphas,
                                  'beta': betas})
        print(arm_pulls.sort_values(by='arm_pulls', ascending=False))
    
    return last_action
