import pandas as pd
import numpy as np
def FindRooms(filename='surgeries.csv'):
    surgeries = pd.read_csv('surgeries.csv')
    start = pd.to_datetime(surgeries.start) - pd.to_datetime(surgeries.start).dt.floor('d')
    end = pd.to_datetime(surgeries.end) - pd.to_datetime(surgeries.end).dt.floor('d')
    start = start.dt.total_seconds().div(900)
    end = end.dt.total_seconds().div(900)
    startOfDay = min(start)
    start-=startOfDay
    end-=startOfDay
    # converting time slot to numbers, since we have 15 minutes time slots, we divide by 900,
    # the first slot is at 7:00 so if we have a surgery 7:15 - 8:00 it will be converted to [1,2,3]
    # and if we have a surgery 7:00 - 8:15 it will be converted to [0,1,2,3,4]

    list = [np.arange(s,e) for s,e in zip(start,end)]
    list = [[idx, l ] for idx,l in enumerate(list)]
    class Room:
        def __init__(self):
            self.listOfHours = []
            self.listOfSurgeryId = []
        def insert(self, Hours,id):
            flattened_list = [item for sublist in self.listOfHours for item in sublist]
            if set(Hours) & set(flattened_list):
                return False
            else:
                self.listOfHours.append(Hours)
                self.listOfSurgeryId.append(id)
                return True
        def deleteLastOne(self):
            self.listOfHours.pop()
            self.listOfSurgeryId.pop()
    rooms = [Room() for i in range(20)]

    list = [l[1] for l in list]
    listIdx = 0
    while True:
        hours = list[listIdx]
        Success = False
        for room in rooms:
            if room.insert(hours,listIdx):
                listIdx+=1
                Success = True
                break
        if listIdx == len(list):
            break
    dictOfIds = {}
    for i,room in enumerate(rooms):
        for j in room.listOfSurgeryId:
            dictOfIds[j] = i
    # sort dict by keys:
    dictOfIds = dict(sorted(dictOfIds.items()))
    return dictOfIds

