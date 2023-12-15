import pandas as pd
import numpy as np
import copy
from recursiveCheck import RecursiveCheck
def calcCostOfFlattened (flattened_list): # calculate the cost of anesthetist
    duration = (max(flattened_list) - min(flattened_list) + 1)/4
    cost = max(5, duration) + 0.5 * max(0, duration - 9)
    return cost
class Anesthetist:
    def __init__(self):
        self.listOfHours = []
        self.listOfSurgeryId = []
    def marry(self,anotherAnesthetist):  # merge two anesthetists to one
        self.listOfHours.extend(anotherAnesthetist.listOfHours)
        self.listOfSurgeryId.extend(anotherAnesthetist.listOfSurgeryId)
    def isMatchPossible(self,anotherAnesthetist):  # check if two anesthetists can be merged
        flattened_list = [item for sublist in self.listOfHours for item in sublist]
        anotherFlattened_list = [item for sublist in anotherAnesthetist.listOfHours for item in sublist]
        if set(flattened_list) & set(anotherFlattened_list):  # if they have common hours
            return False
        if max(flattened_list+anotherFlattened_list) - min(flattened_list+anotherFlattened_list) + 1 > 12 * 4: # if the merged anesthetist will work more than 12 hours
            return False
        costA = self.CalcCost()
        costB = anotherAnesthetist.CalcCost()
        duration = max(flattened_list+anotherFlattened_list) - min(flattened_list+anotherFlattened_list) + 1
        duration /= 4
        costA_B = max(5, duration) + 0.5 * max(0, duration - 9)
        if costA_B > costA + costB: # if the merged anesthetist will cost more than the sum of the two anesthetists
            return False
        return True # if the two anesthetists can be merged and it will be cheaper
    def insert(self,Hours, id):
        flattened_list = [item for sublist in self.listOfHours for item in sublist]
        if set(Hours) & set(flattened_list): #
            return False
        else:
            flattened_list.extend(Hours)
            if max(flattened_list) - min(flattened_list) + 1 > 12 * 4: # if the anesthetist will work more than 12 hours
                return False
            else:
                self.listOfHours.append(Hours)
                self.listOfSurgeryId.append(id)
                return True
    def CalcCost(self): # calculate the cost of anesthetist
        flattened_list = [item for sublist in self.listOfHours for item in sublist]
        duration = max(flattened_list) - min(flattened_list) + 1
        duration /= 4
        cost = max(5, duration) + 0.5 * max(0, duration - 9)
        return cost
    def costOfAdding(self,hours): # calculate the cost of adding a surgery to anesthetist
        listOfHoursAfter = self.listOfHours+[hours]
        flattened_list_before = [item for sublist in self.listOfHours for item in sublist]
        flattened_list_after = [item for sublist in listOfHoursAfter for item in sublist]
        if len(flattened_list_after) != len(np.unique(flattened_list_after)):
            return 1e9
        return calcCostOfFlattened(flattened_list_after)-calcCostOfFlattened(flattened_list_before)

def getTwoDirections():
    surgeries = pd.read_csv('surgeries.csv')
    start = pd.to_datetime(surgeries.start) - pd.to_datetime(surgeries.start).dt.floor('d')
    end = pd.to_datetime(surgeries.end) - pd.to_datetime(surgeries.end).dt.floor('d')
    start = start.dt.total_seconds().div(900)
    end = end.dt.total_seconds().div(900)
    startOfDay = min(start)
    start -= startOfDay
    end -= startOfDay
    # converting time slot to numbers, since we have 15 minutes time slots, we divide by 900,
    # the first slot is at 7:00 so if we have a surgery 7:15 - 8:00 it will be converted to [1,2,3]
    # and if we have a surgery 7:00 - 8:15 it will be converted to [0,1,2,3,4]

    list = [np.arange(s, e) for s, e in zip(start, end)]
    list = [[idx, l] for idx, l in enumerate(list)]

    mid = len(list) // 2
    listA = list[:mid]  # for left to right 7:00 ->
    listB = list[mid:]  # for right to left <- 21:00
    listA.sort(key=lambda x: (x[1][0]), reverse=False)
    listB.sort(key=lambda x: (x[1][-1]), reverse=True)
    idxsA = [l[0] for l in listA]
    idxsB = [l[0] for l in listB]
    listA = [l[1] for l in listA]
    listB = [l[1] for l in listB]


    costToOpenNewAnesthetist = .25
    Anesthetists = []
    for listIdx,hours in enumerate(listA):
        Success = False
        best = 1e9
        bestIdx = -1
        for i,anesthetist in enumerate(Anesthetists):
            costOfAdding = anesthetist.costOfAdding(hours)
            if costOfAdding < best:
                best = costOfAdding
                bestIdx = i
        if bestIdx <0 or best > len(hours)/4 + costToOpenNewAnesthetist :
            Anesthetists.append(Anesthetist())
            f = Anesthetists[-1].insert(hours,idxsA[listIdx])
        else:
            f = Anesthetists[bestIdx].insert(hours,idxsA[listIdx])

    AnesthetistsB = []
    for listIdx,hours in enumerate(listB):
        Success = False
        best = 1e9
        bestIdx = -1
        for i,anesthetist in enumerate(AnesthetistsB):
            costOfAdding = anesthetist.costOfAdding(hours)
            if costOfAdding < best:
                best = costOfAdding
                bestIdx = i

        if bestIdx < 0 or best > len(hours)/4 + costToOpenNewAnesthetist:
            AnesthetistsB.append(Anesthetist())
            f = AnesthetistsB[-1].insert(hours,idxsB[listIdx])
        else:
            f = AnesthetistsB[bestIdx].insert(hours,idxsB[listIdx])
    Anesthetists.extend(AnesthetistsB)
    idxs = idxsA + idxsB

    possibleMatches = []
    for i in range(len(Anesthetists)):
        for j in range(i,len(Anesthetists)):
            if Anesthetists[i].isMatchPossible(Anesthetists[j]):
                possibleMatches.append([i,j])
    print(f"number of possible matches: {len(possibleMatches)}")
    print(possibleMatches)
    ans = RecursiveCheck(possibleMatches)  # the recursive check return all possible matches that don't have  common Anesthetists
    print(f"number of possible solutions: {len(ans)}")
    MIN = 1e9
    for matches in ans:
        tempAnesthetists = copy.deepcopy(Anesthetists)
        for match in matches:
            tempAnesthetists[match[0]].marry(tempAnesthetists[match[1]])
            tempAnesthetists[match[1]] = None
        tempAnesthetists = [anes for anes in tempAnesthetists if anes is not None]
        # calc cost:
        cost = 0
        for anes in tempAnesthetists:
            cost += anes.CalcCost()
        if cost < MIN:
            MIN = cost
            best = matches
            keep = tempAnesthetists

    dictOfIdsAnesthetistBefore = {}
    for i,anes in enumerate(Anesthetists):
        for j in anes.listOfSurgeryId:
            dictOfIdsAnesthetistBefore[j] = i
    dictOfIdsAnesthetistBefore = dict(sorted(dictOfIdsAnesthetistBefore.items()))

    dictOfIdsAnesthetistAfter = {}
    for i,anes in enumerate(keep):
        for j in anes.listOfSurgeryId:
            dictOfIdsAnesthetistAfter[j] = i
    dictOfIdsAnesthetistAfter = dict(sorted(dictOfIdsAnesthetistAfter.items()))
    # sort dict by keys:
    totalCost = 0
    for anes in Anesthetists:
        totalCost += anes.CalcCost()
    print(f"total cost before merging: {totalCost}")
    print(f"Total cost after merging: {MIN}: merging: {best}")
    return dictOfIdsAnesthetistBefore,dictOfIdsAnesthetistAfter

if __name__ == '__main__':
    dictOfIdsAnesthetist = getTwoDirections()