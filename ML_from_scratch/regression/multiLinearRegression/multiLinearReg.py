import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




#DATA PREPROCESSING  AND LOOOOKING INTO IT-----------------------------------------------------------------




data=pd.read_csv("advertising.csv")

CatCol=data.select_dtypes(include=["object"]).columns.tolist()
if(len(CatCol)>0):
    data=pd.get_dummies(data,columns=CatCol)


# print(data.head())

for i in data.columns:
    data[i]=(data[i]-data[i].mean())/data[i].std()






# LOSS FUNCITON CALCULATION ---------------------------------------------------------------------

# x1    x2    x3     x4     y     AND                     yi=m1xi1 + m2xi2 +m3+xi3 + m4xi4 + b




def lossFunction(m,b,xTrain,yTrain):

    E=0
    for i in range(len(xTrain)):
        yPred=b
        for j in range(xTrain.shape[1]):
            yPred+=m[j]*xTrain.iloc[i,j]
        E+=(yTrain.iloc[i]-yPred)**2
    return E/len(xTrain)




#CALCULATE GRADIENT DESCENT --------------------------------------------------------------------------




def GradientDescent(m,b,l,xTrain,yTrain):

    mNew=np.zeros(len(m))
    bNew=0
    n=len(xTrain)


    for i in range(n):
        yPred=b
        for j in range(xTrain.shape[1]):
            yPred+=m[j]*xTrain.iloc[i,j]
        dev=(yTrain.iloc[i]-yPred)
        bNew+=(-2/n)*dev
        for j in range(xTrain.shape[1]):
            mNew[j]+=(-2/n)*xTrain.iloc[i,j]*dev

    b-=l*bNew
    for i in range(len(m)):
        m[i]-=l*mNew[i]

    return m,b




#SPLITTTINGGGGG AND PLOTTTTINGGG-------------------------------------------------------------------------------------


x=data.drop("charges",axis=1)
y=data["charges"]

# features=x.columns
# colours=["red","blue","green"]

# plt.figure(figsize=(8,6))

# for i,col in enumerate(features):
#     plt.scatter(x[col],y,color=colours[i],label=col,alpha=0.7)

# plt.xlabel("X Features Normalized")
# plt.ylabel("Sales")
# plt.title("All Features vs Sales")
# plt.legend()
# plt.show()


xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=42)



#MAIN--------------------------------------------------------------------------------------------------


    
m=np.zeros(xTrain.shape[1])
b=0

print(lossFunction(m,b,xTrain,yTrain))

l,epochs=0.3,100

for i in range(epochs):
    m,b=GradientDescent(m,b,l,xTrain,yTrain)
    if(i%2==0):
        print(f"Epoch {i}: Error- {lossFunction(m,b,xTrain,yTrain)}")


print(lossFunction(m,b,xTest,yTest))




#----------------------GPT VISUALISATION ------------------------

# yPred = []
# for i in range(len(xTest)):
#     pred = b + sum(m[j] * xTest.iloc[i, j] for j in range(xTest.shape[1]))
#     yPred.append(pred)

# # -------------------- DENORMALIZE --------------------
# # Convert normalized values back to original charges
# yTest_real = np.array(yTest) * y.std() + y.mean()
# yPred_real = np.array(yPred) * y.std() + y.mean()

# # Compute MSE and RMSE in original units
# mse_real = ((yTest_real - yPred_real) ** 2).mean()
# rmse_real = np.sqrt(mse_real)
# print("MSE in original charges:", mse_real)
# print("RMSE in original charges:", rmse_real)

# # -------------------- PLOT --------------------
# plt.figure(figsize=(8,6))
# plt.scatter(yTest_real, yPred_real, color='blue', alpha=0.7)
# plt.plot([min(yTest_real), max(yTest_real)], 
#          [min(yTest_real), max(yTest_real)], 
#          color='red', linestyle='--', label='Perfect Prediction')
# plt.xlabel("Actual Charges")
# plt.ylabel("Predicted Charges")
# plt.title("Predicted vs Actual Charges (Original Units)")
# plt.legend()
# plt.show()