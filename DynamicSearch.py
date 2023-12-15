import pandas as pd
import numpy as np
from time import time
dictOfIdsAnesthetist = {}
def calcCostOfFlattened (flattened_list):
    duration = (max(flattened_list) - min(flattened_list) + 1)/4
    cost = max(5, duration) + 0.5 * max(0, duration - 9)
    return cost
class Anesthetist:
    def __init__(self):
        self.listOfHours = []
        self.listOfSurgeryId = []
    def calcAddingCost(self, hours):
        gap = 0.125
        if self.isEmpty():
            return len(hours)/4 +gap
        listOfHoursAfter = self.listOfHours+[hours]
        flattened_list_before = [item for sublist in self.listOfHours for item in sublist]
        flattened_list_after = [item for sublist in listOfHoursAfter for item in sublist]
        if len(flattened_list_after) != len(np.unique(flattened_list_after)):
            return 1e9
        if max(flattened_list_after) - min(flattened_list_after) + 1 > 12 * 4:
            return 1e9
        return calcCostOfFlattened(flattened_list_after)-calcCostOfFlattened(flattened_list_before)

    def CalcCost(self):
        if self.isEmpty():
            return 0
        flattened_list = [item for sublist in self.listOfHours for item in sublist]
        duration = max(flattened_list) - min(flattened_list) + 1
        duration /= 4
        cost = max(5, duration) + 0.5 * max(0, duration - 9)
        return cost
    def insert(self, Hours,id):
        flattened_list = [item for sublist in self.listOfHours for item in sublist]
        if set(Hours) & set(flattened_list):
            return False
        else:
            flattened_list.extend(Hours)
            if max(flattened_list) - min(flattened_list) + 1 > 12 * 4:
                return False
            else:
                self.listOfHours.append(Hours)
                self.listOfSurgeryId.append(id)
                return True
    def removeLast(self):
        self.listOfHours.pop()
        self.listOfSurgeryId.pop()
    def isEmpty(self):
        return len(self.listOfHours) == 0


class ListOfAnesthetists:
    def __init__(self,maxNumOfAnesthetists):
        self.listOfAnesthetists = [Anesthetist() for i in range(maxNumOfAnesthetists)]
    def CalcCost(self):
        return np.sum([anesthetist.CalcCost() for anesthetist in self.listOfAnesthetists])

    def getOrderOfAnesthetist(self,Hours):
        costs = [1e9 for i in range(len(self.listOfAnesthetists))]
        for idx,anesthetist in enumerate(self.listOfAnesthetists):
            costs[idx] = self.listOfAnesthetists[idx].calcAddingCost(Hours)
            if self.listOfAnesthetists[idx].isEmpty():
                break
        # from cost get order indices while ignoring 1e9 values:
        order = np.argsort(costs)
        Relevants = np.sum(np.array(costs) < 1e9)
        return order[:Relevants]


all_costs = np.ones(1000000) * 500
idx_costs = 0
min_costs = 1e9
def RecursiveScheduling(ListOfAnesthetists, listOfSurgeries,T,runTimeMax = 15,FirstIdx=0):
    if time() - T > runTimeMax:
        return
    cost = ListOfAnesthetists.CalcCost()
    global min_costs
    if cost >= min_costs:
        return
    if len(listOfSurgeries) == 0:
        # calculate cost:

        global all_costs
        global idx_costs
        all_costs[idx_costs] = cost
        idx_costs += 1
        if cost < min_costs:
            min_costs = cost
            print(min_costs)
            global dictOfIdsAnesthetist
            dictOfIdsAnesthetist = {}
            for i, anes in enumerate(ListOfAnesthetists.listOfAnesthetists):
                for j in anes.listOfSurgeryId:
                    dictOfIdsAnesthetist[j] = i
            dictOfIdsAnesthetist = dict(sorted(dictOfIdsAnesthetist.items()))
        return
    order = ListOfAnesthetists.getOrderOfAnesthetist(listOfSurgeries[0])

    for idx in order:
        ListOfAnesthetists.listOfAnesthetists[idx].insert(listOfSurgeries[0],FirstIdx)
        RecursiveScheduling(ListOfAnesthetists, listOfSurgeries[1:],T,runTimeMax,FirstIdx+1)
        ListOfAnesthetists.listOfAnesthetists[idx].removeLast()

def DynamicSearch():
    surgeries = pd.read_csv('surgeries.csv')
    start = pd.to_datetime(surgeries.start) - pd.to_datetime(surgeries.start).dt.floor('d')
    end = pd.to_datetime(surgeries.end) - pd.to_datetime(surgeries.end).dt.floor('d')
    start = start.dt.total_seconds().div(900)
    end = end.dt.total_seconds().div(900)
    startOfDay = min(start)
    start -= startOfDay
    end -= startOfDay
    listOfSurgeries = [list(np.arange(s, e)) for s, e in zip(start, end)]
    ListOfAnesthetists_ = ListOfAnesthetists(114)
    RecursiveScheduling(ListOfAnesthetists_,listOfSurgeries,time(), runTimeMax=7)


    return dictOfIdsAnesthetist
