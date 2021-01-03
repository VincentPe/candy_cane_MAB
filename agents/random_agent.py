
import random

# observation.reward
# observation.lastActions
# observation.step
# observation.remainingOverageTime
# observation.agentIndex

# configuration.banditCount
# configuration.episodeSteps
# configuration.actTimeout
# configuration.runTimeout
# configuration.decayRate
# configuration.sampleResolution

def random_agent(observation, configuration):
    
#     print(f'observation contains: {observation}. '
#           f'configuration contains: {configuration}')
    
    return random.randrange(configuration.banditCount)
