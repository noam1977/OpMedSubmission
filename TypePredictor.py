import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
class TypePredictor:
    def __init__(self, dataframe):
        """
        Initialize the predictor with the dataframe.
        """
        self.group_means = dataframe.groupby(['Surgery Type', 'Anesthesia Type'])['Duration in Minutes'].mean()


    def predict(self, typeA, typeB):
        """
        Predict the mean for the given typeA and typeB.
        """
        if (typeA, typeB) in self.group_means.index:
            return self.group_means.loc[(typeA, typeB)]
        else:
            return "Group not found in the data"

class KnnTypePredictor:
    def __init__(self, dataframe,n_neighbors):
        self.n_neighbors = n_neighbors
        self.dictOfKnns = {}
        self.normalization_dict = {}
        surgeries = np.unique(dataframe['Surgery Type'])
        anesthesias = np.unique(dataframe['Anesthesia Type'])
        dataFrameGrouped = dataframe.groupby(['Surgery Type', 'Anesthesia Type'])
        for s in surgeries:
            for a in anesthesias:
                dataFrameGrouped.get_group((s, a))
                self.dictOfKnns[(s, a)] = KNeighborsRegressor(n_neighbors=self.n_neighbors)
                x = np.array(dataFrameGrouped.get_group((s, a))[['Age', 'BMI']])
                # normlize Age and BMI:
                mean0 = np.mean(x[:, 0])
                mean1 = np.mean(x[:, 1])
                std0 = np.std(x[:, 0])
                std1 = np.std(x[:, 1])
                x[:, 0] = (x[:, 0] - mean0) / std0
                x[:, 1] = (x[:, 1] - mean1) / std1
                self.normalization_dict[(s, a)] = [mean0, mean1, std0, std1]

                y = np.array(dataFrameGrouped.get_group((s, a))['Duration in Minutes'])
                self.dictOfKnns[(s, a)].fit(x, y)


    def predict(self,df):

        type_A = df['Surgery Type']
        type_B = df['Anesthesia Type']
        x = np.array([df[['Age', 'BMI']]])
        # normlize Age and BMI:
        mean0 = self.normalization_dict[(type_A, type_B)][0]
        mean1 = self.normalization_dict[(type_A, type_B)][1]
        std0 = self.normalization_dict[(type_A, type_B)][2]
        std1 = self.normalization_dict[(type_A, type_B)][3]
        x[:, 0] = (x[:, 0] - mean0) / std0
        x[:, 1] = (x[:, 1] - mean1) / std1

        if (type_A, type_B) in self.dictOfKnns:
              return self.dictOfKnns[(type_A, type_B)].predict(x)

        else:
            return "Group not found in the data"

class KnnTypePredictorAgeOnly:
    def __init__(self, dataframe,n_neighbors):
        self.n_neighbors = n_neighbors
        self.dictOfKnns = {}
        surgeries = np.unique(dataframe['Surgery Type'])
        anesthesias = np.unique(dataframe['Anesthesia Type'])
        dataFrameGrouped = dataframe.groupby(['Surgery Type', 'Anesthesia Type'])
        for s in surgeries:
            for a in anesthesias:
                dataFrameGrouped.get_group((s, a))
                self.dictOfKnns[(s, a)] = KNeighborsRegressor(n_neighbors=self.n_neighbors)
                x = np.array(dataFrameGrouped.get_group((s, a))[['Age']])
                y = np.array(dataFrameGrouped.get_group((s, a))['Duration in Minutes'])
                self.dictOfKnns[(s, a)].fit(x, y)


    def predict(self,df):

        type_A = df['Surgery Type']
        type_B = df['Anesthesia Type']
        x = np.array([df[['Age']]])

        if (type_A, type_B) in self.dictOfKnns:
              return self.dictOfKnns[(type_A, type_B)].predict(x)

        else:
            return "Group not found in the data"
class PolyTypePredictor:
    def __init__(self, dataframe,deg = 2):
        self.dictOfPolyPredictors = {}
        surgeries = np.unique(dataframe['Surgery Type'])
        anesthesias = np.unique(dataframe['Anesthesia Type'])
        dataFrameGrouped = dataframe.groupby(['Surgery Type', 'Anesthesia Type'])
        for s in surgeries:
            for a in anesthesias:
                dataFrameGrouped.get_group((s, a))
                # create a polynom 2 degree for each group:
                self.dictOfPolyPredictors[(s, a)] = np.poly1d(np.polyfit(dataFrameGrouped.get_group((s, a))['Age'],
                                                                         dataFrameGrouped.get_group((s, a))[
                                                                             'Duration in Minutes'], deg=deg))

    def predict(self,df):

        type_A = df['Surgery Type']
        type_B = df['Anesthesia Type']

        if (type_A, type_B) in self.dictOfPolyPredictors:
              return self.dictOfPolyPredictors[(type_A, type_B)](df['Age'])

        else:
            return "Group not found in the data"