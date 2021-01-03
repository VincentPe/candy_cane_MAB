# Candy Cane MAB
The multi arm bandit Kaggle christmas special is about getting the highest score from
100 vending machines with different reward distributions. 

You play against a different opponent with a similar ranking each time and 
have to try to get a higher score than the person you are up against in order 
to climb in the ranking.
Each participant gets 2000 tries per episode. The distributions are different per episode,
and every time a vending machine is chosen, the likelihood of a reward decreases by 3% for
that specific machine. Therefore, it's important in the exploration vs exploitation trade 
off to get rewards fast in the episode before the opponent is milking the 'good machines' dry.

I joined this competition to try out a couple of MAB approaches I had learned from the
RL specialization I took. There is not really a delayed reward structure involved in which
you have to plan ahead, nor does function approximation involve complex algorithms since 
the distributions are made at random. It was interesting to see though that game theory played
quite a large role. 

