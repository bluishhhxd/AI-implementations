import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# np.random.seed(42)

# n=500

# x1=np.random.randint(-50,50,n)
# x2=np.random.randint(-20,40,n)
# x3=np.random.randint(-40,20,n)


# noise=np.random.normal(0,30,n)
# y=(3*x1**2 + 2*x2**3 + 0.5*x3 + 0.8*x1*x2 - 1.2*x2*x3 + 0.5*(x1**2)*x3 + 10 + noise) 
# data=pd.DataFrame({"X1":x1,"X2":x2,"X3":x3,"Y":y})

data=pd.read_csv("Ice_cream.csv")


for i in data.columns:
    data[i]=(data[i]-data[i].mean())/data[i].std()


# for i in range(len(data)):
#     # for j in range(len(data.iloc[i])):
#     #     # print(j)
#     #     print(str(data.iloc[i,j])+" ",end="")
#     # print()    
#     print(data.iloc[i])

# sns.pairplot(data)
# plt.show()


#---------------------------RECURSIVE FUNCTION TO GENERATE POLYNOMIAL FEATURES-------------------------------------------

def solve(curDegree,prevColName,prevCol,Ind,degree,x,polyData):

    if(curDegree==degree):
        return
    
    for i in range(Ind,len(x.columns)):
        curCol=x[x.columns[i]]*prevCol
        polyData[prevColName+x.columns[i]]=curCol
        solve(curDegree+1,prevColName+x.columns[i],curCol,i,degree,x,polyData)


def getPolyData(x,degree):

    polyData=pd.DataFrame()
    for i in range(len(x.columns)):
        curCol=x[x.columns[i]]
        polyData[x.columns[i]]=curCol
        solve(1,x.columns[i],curCol,i,degree,x,polyData)
    
    return polyData


#---------------------------LOSS AND GRADIENT DESCENT OPIMIZATION-------------------------------------------------------
        
    
def LossFunction(m,b,xTrain,yTrain):
    n=len(xTrain)
    sumError=0
    for i in range(len(xTrain)):
        yPred=b
        for j in range(len(xTrain.iloc[i])):                  # Looping over Columns of all the feature/independent Variables
            yPred+=m[j]*xTrain.iloc[i,j]
        sumError+=(yTrain.iloc[i]-yPred)**2                   # Summing Squared (Yactual-Ypredicted)
    return sumError/n



def GradientDescent(m,b,l,l1,l2,xTrain,yTrain):
    n=len(xTrain)
    mNew=np.zeros(len(m))                   # Array of Slope Directions to add  in our slopes
    bNew=0                                  # New Direction to add in our intercept
            
    for i in range(len(xTrain)):
        yPred=b               
        for j in range(len(xTrain.iloc[i])):                    #This loop Calculates Ypred
            yPred+=m[j]*xTrain.iloc[i,j]
        dev=yTrain.iloc[i]-yPred
        bNew+=(-2/n)*dev
        for j in range(len(xTrain.iloc[i])):                   #This loop updates our Array of Slope Directions
            mNew[j]+=(-2/n)*xTrain.iloc[i,j]*dev

    b-=l*bNew
    for i in range(len(m)):                                     # 2*l2*m[i] is the penalty but in derivative form for Ridge
        m[i]-=l*(mNew[i]+2*l2*m[i]+l1*np.sign(m[i]))                          #l1*sign of m[i] here is the derivative form for Lasso 
    return m,b
        
    


    


x=data.drop("Ice-Cream Sales",axis=1)
y=data["Ice-Cream Sales"]

x=getPolyData(x,2)
xTrain,xTest,yTrain,yTest=train_test_split(x,y,random_state=23,train_size=0.8)

m=np.zeros(len(xTrain.columns))
b=0
l=0.01

# print(LossFunction(m,b,xTrain,yTrain))

epochs=2000

for i in range(epochs):
    m,b=GradientDescent(m,b,l,0,0,xTrain,yTrain)
    if(i%10==0):
        print(f"Epoch {i}:- {LossFunction(m,b,xTrain,yTrain)}")

print(m)
print(b)

print(LossFunction(m,b,xTest,yTest))



#---------------------PLOTTING AND DENORMALIZING-------------------------------------

# m=[0.04506478,1.09393749]
# b=-1.0889929438791255

# yPred = []
# for i in range(len(xTest)):
#     pred = b + sum(m[j] * xTest.iloc[i, j] for j in range(xTest.shape[1]))
#     yPred.append(pred)

# # Convert normalized values back to original charges
# yTest_real = np.array(yTest) * y.std() + y.mean()
# yPred_real = np.array(yPred) * y.std() + y.mean()

# # Compute MSE and RMSE in original units
# mse_real = ((yTest_real - yPred_real) ** 2).mean()
# rmse_real = np.sqrt(mse_real)
# print("MSE in original charges:", mse_real)
# print("RMSE in original charges:", rmse_real)




