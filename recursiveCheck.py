def recursiveCheck(listOfMatches,ANS,listOfCurrentMatches=[],idx=0):
    flattend = [item for sublist in listOfCurrentMatches for item in sublist]
    if len(flattend) != len(set(flattend)):
        return
    if idx == len(listOfMatches):
        ANS.append(listOfCurrentMatches)
        return

    recursiveCheck(listOfMatches, ANS,listOfCurrentMatches.copy(), idx + 1)
    listOfCurrentMatches.append(listOfMatches[idx])
    recursiveCheck(listOfMatches, ANS,listOfCurrentMatches.copy(), idx + 1)


def RecursiveCheck(listOfMatches):
    ANS = []
    recursiveCheck(listOfMatches, ANS)
    return ANS
if __name__ == "__main__":
    listOfMatches = [[0,1],[2,3],[3,4]]
    print(RecursiveCheck(listOfMatches))