import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# States Definition

wind_dir = np.linspace(-90,80,18)
Nacelle_ang = np.linspace(-90,80,18)
actions = np.array([-10,0,+10])

# R matrix

A = np.array(list(itertools.product(wind_dir, Nacelle_ang,actions)))
table = pd.DataFrame({'wind_dir': A[:,0],'Nacelle_ang': A[:,1], 'action': A[:,2]})

table.drop(table[(table['wind_dir'] - table['Nacelle_ang'] == -90) & (table['action'] == 10)].index, inplace = True)
table.drop(table[(table['wind_dir'] - table['Nacelle_ang'] ==  90) & (table['action'] == -10)].index, inplace = True)


def rewardFunc(row):
    return np.cos((row['wind_dir'] - (row['Nacelle_ang'] + row['action']))*np.pi/180)


def statePropagation(state, action):
    return(state[0] , state[1] + action)
    
    

table['reward'] = table.apply(lambda row: rewardFunc(row), axis = 1)

R_table = pd.pivot_table(table, values='reward', index=['wind_dir' , 'Nacelle_ang'],
                    columns=['action'], aggfunc=np.sum)

R = R_table.values

# Q matrix
Q = pd.DataFrame(0, index = R_table.index, columns = R_table.columns)

# Gamma (learning parameter).
gamma = 0.4
alpha = 0.9

def available_actions(state):
    current_state_row = R_table.loc[state]
    av_act = ~np.isnan(current_state_row.values)
    av_act = actions[av_act]
    return av_act.astype(int)

for i in range(20000):
    current_state = np.random.choice(R_table.index,1, replace = False)
    available_act = available_actions(current_state[0])
     
    action = np.random.choice(available_act)
    # print(action)
    next_state = statePropagation(current_state[0] ,action)
    Q_next_max = np.max(Q.loc[current_state].values)
    # rewards = R_table.loc[current_state[0]]

    Q.loc[current_state, action] = Q.loc[current_state, action] + alpha * (R_table.loc[current_state,action] + gamma * Q_next_max - Q.loc[current_state, action])
    

states = [(-90, 80)]
episodes = range(20)
action_sequence = []

for i in episodes:

    next_step_index = np.where(Q.loc[states[-1]].values == np.max(Q.loc[states[-1]].values))[0]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    av_act = actions[next_step_index]
    
    next_state = statePropagation(states[-1],av_act)
    states.append(next_state)
    action_sequence.append(av_act)

print('State Sequence:        ')
print(states)




fig, ax = plt.subplots(2,1)
ax[0].step(episodes, [s[0] - s[1] for s in states[:-1]])
ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_title('Yaw Misalignment vs Steps')
ax[0].set_xlabel('Episode Number')
ax[0].set_ylabel('Yaw Misalignment (deg)')

ax[1].step(episodes, action_sequence)
ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set_title('Actions vs Steps')
ax[1].set_xlabel('Episode Number')
ax[1].set_ylabel('Action (deg)')
