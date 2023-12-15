from plot_day_schedule import plot_day_schedule
from matplotlib import pyplot as plt
from twoDirections import getTwoDirections
from FindRooms import FindRooms
import pandas as pd
from DynamicSearch import DynamicSearch
dictOfIdsAnesthetistBefore,dictOfIdsAnesthetistAfter = getTwoDirections()
dictOfIds = FindRooms()
df = pd.read_csv('surgeries.csv')
df.rename(columns={'Unnamed: 0': 'example_sol'}, inplace=True)
df.rename(columns={'start': 'start_time'}, inplace=True)
df.rename(columns={'end': 'end_time'}, inplace=True)
df['room_id'] = ['room - ' + str(i) for i in dictOfIds.values()]


df['anesthetist_id'] = ['anesthetist - ' + str(i) for i in dictOfIdsAnesthetistBefore.values()]
plot_day_schedule(df)
plt.title('Before merging')

df['anesthetist_id'] = ['anesthetist - ' + str(i) for i in dictOfIdsAnesthetistAfter.values()]
plot_day_schedule(df)
plt.title('After merging')

dictOfIds = DynamicSearch()
df['anesthetist_id'] = ['anesthetist - ' + str(i) for i in dictOfIds.values()]
plot_day_schedule(df)
plt.title('Dynamic Search')

plt.show()
