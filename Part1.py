import pandas
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from TypePredictor import KnnTypePredictor, PolyTypePredictor, KnnTypePredictorAgeOnly
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures


pd = pandas.read_csv("surgeries to predict.csv")

X = pd[['Surgery Type', 'Anesthesia Type', 'Age', 'BMI']]
y = pd['Duration in Minutes']
predictions = np.ones(len(y)) * np.mean(y)
rmse = np.sqrt(mean_squared_error(y, predictions))
print(f"RMSE where prediction is the average value: {rmse}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # i want the test to be always the same
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

# make one dataFrame from X_train and y_train:
train = X_train.copy()
train['Duration in Minutes'] = y_train

from TypePredictor import TypePredictor
TypePredictorModel = TypePredictor(train)
predictions = []
for i in range(len(X_val)):
    predictions.append(TypePredictorModel.predict(X_val.iloc[i]['Surgery Type'], X_val.iloc[i]['Anesthesia Type']))
rmse = np.sqrt(mean_squared_error(y_val, predictions))
print(f"RMSE - prediction is according to average value of group (group is with the same Surgery Type and Anesthesia Type): {rmse}")

MIN = 1e9
for n_neighbors in tqdm(range(1,120,4)):
    KnnTypePredictorModel = KnnTypePredictor(train, n_neighbors)
    predictions = []
    for i in range(len(X_val)):
        predictions.append(KnnTypePredictorModel.predict(X_val.iloc[i]))
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    # save the best model and its n_neighbors
    if rmse < MIN:
        bestN_neighbors = n_neighbors
        MIN = rmse
print(f"RMSE (using Knn with {bestN_neighbors} Neighbors): ", MIN)

MIN = 1e9
# use tqdm to see the progress of the loop:
for n_neighbors in tqdm(range(1,120,4)):
    KnnTypePredictorModel = KnnTypePredictorAgeOnly(train, n_neighbors)
    predictions = []
    for i in range(len(X_val)):
        predictions.append(KnnTypePredictorModel.predict(X_val.iloc[i]))
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    # save the best model and its n_neighbors
    if rmse < MIN:
        bestN_neighbors = n_neighbors
        MIN = rmse
print(f"RMSE (using Knn with {bestN_neighbors} Neighbors (Age Only)): ", MIN)
polypredictors ={}
for ii in range(1,5):
    PolyTypePredictorModel = PolyTypePredictor(train,ii)
    polypredictors[ii] = PolyTypePredictorModel
    predictions = []
    for i in range(len(X_val)):
        predictions.append(PolyTypePredictorModel.predict(X_val.iloc[i]))
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f"RMSE using polfit on age, degree of polynom = {ii} ", rmse)

# OneHotEncoder
encSurgery = OneHotEncoder(handle_unknown='ignore')
encSurgery.fit(X_train[['Surgery Type']])
encAnesthesia = OneHotEncoder(handle_unknown='ignore')
encAnesthesia.fit(X_train[['Anesthesia Type']])
X_train_enc_Surgery= encSurgery.transform(X_train[['Surgery Type']])
X_train_enc_Anesthesia = encAnesthesia.transform(X_train[['Anesthesia Type']])
X_val_enc_Surgery = encSurgery.transform(X_val[['Surgery Type']])
X_val_enc_Anesthesia = encAnesthesia.transform(X_val[['Anesthesia Type']])
X_test_enc_Surgery = encSurgery.transform(X_test[['Surgery Type']])
X_test_enc_Anesthesia = encAnesthesia.transform(X_test[['Anesthesia Type']])
# add the rest of the features:
# z=X_train[['Age', 'BMI']].values -> polynomial features, calc z:

# Assuming df is your DataFrame
poly = PolynomialFeatures(2)

X_train_enc = np.concatenate((X_train_enc_Surgery.toarray(), X_train_enc_Anesthesia.toarray(), poly.fit_transform(X_train[['Age', 'BMI']])), axis=1)
X_val_enc = np.concatenate((X_val_enc_Surgery.toarray(), X_val_enc_Anesthesia.toarray(), poly.fit_transform(X_val[['Age', 'BMI']])), axis=1)
X_test_enc = np.concatenate((X_test_enc_Surgery.toarray(), X_test_enc_Anesthesia.toarray(), poly.fit_transform(X_test[['Age', 'BMI']])), axis=1)

# N = NNmodel()
# N.train(torch.from_numpy(X_train_enc.astype(np.float32)).unsqueeze(0),torch.from_numpy(np.array(y_train).astype(np.float32)),epochs=10000,lr=0.001,batch_size=32)

MIN = 1e9
for n_estimators in tqdm([25,50,100,150,200,250]):
    for learning_rate in [0.0001,0.001,0.01,0.1,1]:
        for max_depth in [1,2,3,4,5,6]:
            GradientBoostingRegressorModel = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=0, loss='squared_error')
            GradientBoostingRegressorModel.fit(X_train_enc, y_train)
            predictions = GradientBoostingRegressorModel.predict(X_val_enc)
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            # print(f"{n_estimators}, {learning_rate}, {max_depth}, RMSE: ", rmse)
            if rmse < MIN:
                bestN_estimators = n_estimators
                bestLearning_rate = learning_rate
                bestMax_depth = max_depth
                MIN = rmse
            MIN = min(MIN, rmse)
print(f"\nRMSE (using GradientBoostingRegressor with {bestN_estimators} n_estimators, {bestLearning_rate} learning_rate, {bestMax_depth} max_depth): ", MIN)
#now using random forest:
MIN = 1e9
for n_estimators in tqdm([25,50,100,150,200,250,300,400]):
    for max_depth in [1,2,3,4,5,6]:
        RandomForestRegressorModel = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        RandomForestRegressorModel.fit(X_train_enc, y_train)
        predictions = RandomForestRegressorModel.predict(X_val_enc)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        if rmse < MIN:
            bestN_estimators = n_estimators
            bestMax_depth = max_depth
            MIN = rmse
        MIN = min(MIN, rmse)
print(f"RMSE (using RandomForestRegressor with {bestN_estimators} n_estimators, {bestMax_depth} max_depth): ", MIN)

predictions = []
for i in range(len(X_val)):
    predictions.append(polypredictors[2].predict(X_test.iloc[i]))
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Test data results:\nRMSE using polfit on age, degree of polynom = 2 ", rmse)