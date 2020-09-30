
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm 
import floris.tools as wfct
import matplotlib

plt.close('all')
#%% Floris

# initialize FLORIS interface
fi = wfct.floris_interface.FlorisInterface("input.json")
fi.reinitialize_flow_field(layout_array=[[0, 0], [0, 200]])      # system parameter

# define farm power generator
def rewardFunc(states, actions = None):

    # changing the wind farm configuration
    wind_direction = states[0]
    nacelle_angles = states[1:]
    yaw_angles = wind_direction - np.array(nacelle_angles)
    
    
    if actions == None:
        actions_ = []
        for i in range(len(states) - 1):
            actions_.append(0)
        actions = tuple(actions_)
    
    
        
    fi.reinitialize_flow_field(wind_direction = wind_direction)

    # calculating the power
    fi.calculate_wake(yaw_angles = yaw_angles)
    power_0 = fi.get_farm_power()
    fi.calculate_wake(yaw_angles = yaw_angles - actions)
    power_1 = fi.get_farm_power()

    # computing the reward
    reward = power_1 - power_0

    
    # plot and visualization
    # hor_plane = fi.get_hor_plane()
    # fig, ax = plt.subplots()
    # wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    # plt.show()
    # new_state = [states[0], states[1] + action]
    new_states = list(np.array(states[1:]) + np.array(actions))
    new_states.insert(0, states[0])
    return pd.Series([reward, tuple(new_states)])



#%%
# States Definition
wind_dir = np.linspace(-6,6,13)
Nacelle_ang = np.linspace(-5,5,11)

statesList = [wind_dir]
for i in range(2):
    statesList.append(Nacelle_ang)  
S = (list(itertools.product(*statesList)))

actionList = []
for i in range(2):
    actionList.append(np.array([-1,0,1]))
U = (list(itertools.product(*actionList)))
    

table = pd.DataFrame(0, index = S, columns = U).stack().reset_index().rename(columns={'level_0':'States','level_1':'Actions',0:'Rewards'})

#Removing Infeasible Action-State Rows
def ActionValidity(row):
    turbine_num = len(row['Actions'])
    for i in range(turbine_num):
        if (((row['States'][i+1] + row['Actions'][i]) > np.max(Nacelle_ang)) | ((row['States'][i+1] + row['Actions'][i]) < np.min(Nacelle_ang))):
            return True
    return False

table['Invalid'] = table.apply(lambda row: ActionValidity(row), axis = 1)
table = table[table['Invalid'] == False].reset_index(drop = True)


# Function to Move to the Next State by the Input Action
def statePropagation(states, action):
    new_states = list(np.array(states[1:]) + np.array(action))
    new_states.insert(0, states[0])
    return tuple(new_states)

# Reward Table 
tqdm.pandas()
table[['Rewards','Next States']] = table.progress_apply(lambda row: rewardFunc(row['States'], row['Actions']), axis = 1)

table['Q'] = 0




#%%  Training Loop to Fill the Q Matrix

# Gamma and Alpha (learning parameter).
gamma = 0.9
alpha = 0.4

for i in tqdm(range(30000)):
    current_state = np.random.choice(table['States'],1, replace = False)[0]
    action = np.random.choice(table[table['States'] == current_state]['Actions'].values,1,replace = False)[0]
    next_state = table[(table['States'] == current_state) & (table['Actions'] == action)]['States'].values[0]
    Q_next_max = np.max( table[table['Next States'] == next_state]['Q'].values)
    table.loc[(table['States'] == current_state) & (table['Actions'] == action),'Q'] = table.loc[(table['States'] == current_state) & (table['Actions'] == action),'Q'] + alpha * (table.loc[(table['States'] == current_state) & (table['Actions'] == action),'Rewards'] + gamma * Q_next_max - table.loc[(table['States'] == current_state) & (table['Actions'] == action),'Q'])
    



#%%  Determining the States and Action Sequence

# Initial State
states = [(0, 4, -3)]
episodes = range(1,20)
action_sequence = []
for i in episodes:    
    BestActionInd = table.loc[(table['States'] == states[-1])]['Q'].idxmax()
    BestAction = table.iloc[BestActionInd]['Actions']
    next_state = table.iloc[BestActionInd]['Next States']
    states.append(next_state)
    action_sequence.append(BestAction)



#%% Post Processing
power = []
yaw1 = [s[0] - s[1] for s in states]
yaw2 = [s[0] - s[2] for s in states]


for i in range(len(states)):
    
    yaw_angles = [yaw1[i],yaw2[i]]
    fi.reinitialize_flow_field(wind_direction = states[i][0])    
    fi.calculate_wake(yaw_angles = yaw_angles)
    power.append(fi.get_farm_power())


print('State Sequence:        ')
print(states)
print('\nAction Sequence:        ')
print(action_sequence)

font = {'size'   : 16}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(1,1)
ax.step(episodes, [s[0] - s[1] for s in states[:-1]])
ax.step(episodes, [s[0] - s[2] for s in states[:-1]])
ax.set_title('Yaw Misalignment vs Episodes')
ax.set_xlabel('Episodes Number')
ax.set_ylabel('Yaw Misalignment (deg)')

fig, ax = plt.subplots(1,1)
ax.step(episodes, action_sequence)
ax.set_title('Actions vs Episodes')
ax.set_xlabel('Episodes Number')
ax.set_ylabel('Action (deg)')

fig, ax = plt.subplots(1,1)
ax.step(episodes, np.array(power[:-1])/1000)
ax.set_title('Farm Power vs Episodes')
ax.set_xlabel('Episodes Number')
ax.set_ylabel('Power (kW)')

fig, ax = plt.subplots(1,2)

fi.reinitialize_flow_field(wind_direction = states[0][0])
fi.calculate_wake(yaw_angles = [yaw1[0],yaw2[0]])
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax[0])

fi.reinitialize_flow_field(wind_direction = states[-1][0])
fi.calculate_wake(yaw_angles = [yaw1[-1],yaw2[-1]])
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax[1])


