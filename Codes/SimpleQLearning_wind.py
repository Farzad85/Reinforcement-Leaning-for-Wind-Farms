
import numpy as np
import pandas as pd
import itertools

# States Definition

wind_dir = np.linspace(-90,90,19)
wind_speed = np.linspace(3,14,10)
Nacelle_ang = np.linspace(-90,90,19)
action = np.array([-10,0,+10])

A = np.array(list(itertools.product(wind_dir, wind_speed, Nacelle_ang,action)))
Q_table = pd.DataFrame({'wind_dir': A[:,0],'wind_speed': A[:,1],'Nacelle_ang': A[:,2], 'action': A[:,3]})

Q_table.drop(Q_table[(Q_table['wind_dir'] - Q_table['Nacelle_ang'] == -90) & (Q_table['action'] == -10)].index, inplace = True)
Q_table.drop(Q_table[(Q_table['wind_dir'] - Q_table['Nacelle_ang'] ==  90) & (Q_table['action'] ==  10)].index, inplace = True)


def rewardFunc(row):
    
    if (np.cos((row['wind_dir'] - row['Nacelle_ang'] + row['action'])*np.pi/180) - np.cos((row['wind_dir'] - row['Nacelle_ang'])*np.pi/180))  > 0:
        return 1
    elif (np.cos((row['wind_dir'] - row['Nacelle_ang'] + row['action'])*np.pi/180) - np.cos((row['wind_dir'] - row['Nacelle_ang'])*np.pi/180)) < 0:
        return -1
    else:
        return 0

Q_table['reward'] = Q_table.apply(lambda row: rewardFunc(row), axis = 1)

table = pd.pivot_table(Q_table, values='reward', index=['wind_dir', 'wind_speed' , 'Nacelle_ang'],
                    columns=['action'], aggfunc=np.sum)



# ((1/(1 + np.exp(-row['wind_speed']*0.6+ 5)))