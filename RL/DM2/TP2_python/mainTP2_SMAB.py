import numpy as np
import arms
import matplotlib.pyplot as plt

# Build your own bandit problem


# this is an example, please change the parameters or arms!
arm1 = arms.ArmBernoulli(0.7, random_state=np.random.randint(1, 312414))
arm2 = arms.ArmBernoulli(0.5, random_state=np.random.randint(1, 312414))
arm3 = arms.ArmBernoulli(0.35, random_state=np.random.randint(1, 312414))
arm4 = arms.ArmBernoulli(0.15, random_state=np.random.randint(1, 312414))

MAB = [arm1, arm2, arm3, arm4]

# bandit : set of arms

nb_arms = len(MAB)
means = [el.mean for el in MAB]

# Display the means of your bandit (to find the best)
print('means: {}'.format(means))
mu_max = np.max(means)
print('mu_max: {}'.format(mu_max))

def UCB1(T, MAB):
    samples = dict()
    lenMab = len(MAB)
    
    rew = []
    draws = []
    
    for i in range(lenMab):
        s = int(MAB[i].sample())
        samples[i] = []
        samples[i].append(s)
        draws.append(i)
        rew.append(s)
    
    c = 0.45
    
    for t in range(lenMab, T):
        A = [computeUCB(samples[i], t, c) for i in range(lenMab)]
        A_t = np.random.choice(np.flatnonzero(A == np.max(A))) # we use randomization in case of equality
        s = int(MAB[A_t].sample())
        samples[A_t].append(s)
        draws.append(A_t)
        rew.append(s)
        
    return rew, draws

def computeUCB(samples, t, c):
    return np.mean(samples) + c * np.sqrt(np.log(T)/(2*len(samples)))

def TS(T, MAB):
    samples = dict()
    lenMab = len(MAB)
    
    for i in range(lenMab):
        samples[i] = []
    
    rew = []
    draws = []
    
    for t in range(T):
        A = [computeTS(samples[i]) for i in range(lenMab)]
        A_t = np.random.choice(np.flatnonzero(A == np.max(A))) # we use randomization in case of equality
        s = int(MAB[A_t].sample())
        samples[A_t].append(s)
        draws.append(A_t)
        rew.append(s)
        
    return rew, draws

def computeTS(samples):
    S_a = np.sum(samples)
    N_a = len(samples)
    return np.random.beta(S_a + 1, N_a - S_a + 1)
        

def Naive(T, MAB):
    samples = dict()
    lenMab = len(MAB)
    
    rew = []
    draws = []
    
    for i in range(lenMab):
        s = int(MAB[i].sample())
        samples[i] = []
        samples[i].append(s)
        draws.append(i)
        rew.append(s)
    
    for t in range(lenMab, T):
        A = [np.mean(samples[i]) for i in range(lenMab)]
        A_t = np.random.choice(np.flatnonzero(A == np.max(A))) # we use randomization in case of equality
        s = int(MAB[A_t].sample())
        samples[A_t].append(s)
        draws.append(A_t)
        rew.append(s)
        
    return rew, draws


# Comparison of the regret on one run of the bandit algorithm
# try to run this multiple times, you should observe different results

T = 5000  # horizon

rew1, draws1 = UCB1(T, MAB)
reg1 = mu_max * np.arange(1, T + 1) - np.cumsum(rew1)
rew2, draws2 = TS(T, MAB)
reg2 = mu_max * np.arange(1, T + 1) - np.cumsum(rew2)


rew3, draws3 = Naive(T, MAB)
reg3 = mu_max * np.arange(1, T + 1) - np.cumsum(rew3)

# add oracle t -> C(p)log(t)

plt.figure(1)
x = np.arange(1, T+1)
plt.plot(x, reg1, label='UCB', color = 'red')
plt.plot(x, reg2, label='Thompson', color = 'blue')
plt.plot(x, reg3, label='Naive', color = 'orange')
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
#
plt.show()

# (Expected) regret curve for UCB and Thompson Sampling
