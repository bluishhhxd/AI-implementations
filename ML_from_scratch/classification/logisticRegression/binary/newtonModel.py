import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



data=pd.read_csv("breast-cancer.csv")
data=data.drop("id",axis=1)

for i,val in data["diagnosis"].items():
    if(val=='M'):
        data.at[i,"diagnosis"]=1
    else:
        data.at[i,"diagnosis"]=0

data["diagnosis"]=data["diagnosis"].astype(int)
    

#----------------SPLITTING AND PREPROCESSING-------------------

x=data.drop("diagnosis",axis=1)
y=data["diagnosis"]

# print(x.head())
# print(y.head())

xTrain,xTest,yTrain,yTest=train_test_split(x,y,random_state=42,train_size=0.7)

for i in xTrain.columns:
    meanTrain=xTrain[i].mean()
    stdTrain=xTrain[i].std()
    xTrain[i]=(xTrain[i]-meanTrain)/stdTrain
    xTest[i]=(xTest[i]-meanTrain)/stdTrain



#----------------------Maximum Likelihood Estimation (MLE) --------------------------------

def lossFunction(m,b,l1,l2):
    nlkd=0
    for i in range(len(xTrain)):
        z=b
        for j in range(len(m)):
            z+=m[j]*xTrain.iloc[i,j]
        e=np.exp(-z)
        hx=1/(1+e)
        eps=1e-15
        hx=np.clip(hx,eps,1-eps)
        nlkd+=(yTrain.iloc[i]*np.log(hx))+(1-yTrain.iloc[i])*(np.log(1-hx))
    ridge=0
    lasso=0
    for j in m:
        ridge+=j*j
        lasso+=abs(j)
    ridge*=l2
    lasso*=l1
    nlkd=(-nlkd+ridge+lasso)/len(xTrain)
    return nlkd



def newtonModel(m,b):

    mGrads=np.zeros(len(m))
    Hessian=np.zeros((len(m),len(m)))
    bGrad=0

    for i in range(len(xTrain)):
        z=b
        for j in range(len(m)):
            z+=m[j]*xTrain.iloc[i,j]
        e=np.exp(-z)
        hx=1/(1+e)
        yActual=yTrain.iloc[i]
        bGrad+=(yActual-hx)/(len(xTrain))
        for j in range(len(m)):
            mGrads[j]+=((yActual-hx)*xTrain.iloc[i,j])/(len(xTrain))
            for k in range(len(m)):
                Hessian[j][k]-=((e/((1+e)**2))*xTrain.iloc[i,j]*xTrain.iloc[i,k])/(len(xTrain))



    # Gauss Jordan elimination to find the Change in parameters that we need to subtract, i.e H*delta(w)=Grad(w), so we are finding delta(w) which is a vector of parameter changes
    
    for c in range(len(m)):
        val=0
        idx=-1                              #Index of row where our pivot lies
        for r in range(c,len(m)):             #Taking the Largest Value in the entire column c as pivot
            if(abs(val)<abs(Hessian[r,c])):
                val=Hessian[r,c]
                idx=r
        if(idx==-1):                        #Matrix is not invertible
            print("Matrix is not invertible, Newton's method can't be applied")
            return
        if(idx!=c):                         #Swapping
            mGrads[idx],mGrads[c]=mGrads[c],mGrads[idx]              #Swapping Gradient Value at idx <-> c
            for col in range(len(m)):      
                Hessian[idx,col],Hessian[c,col]=Hessian[c,col],Hessian[idx,col]          #Swapping  row idx <-> row c, so that we normalize our diagonals to 1, i.e Row swwapping operation
        #Swapping is done now we can perform the operations over row c

        mGrads[c]/=val                #Normalizing the value
        for col in range(len(m)):     #Normalizing our row, Now this diagonal is 1
            Hessian[c,col]/=val
        for r in range(len(m)):       #Making this column's values 0 except the diagonal
            if(r==c):                 #Leaving the Diagonal Column
                continue
            curRowVal=Hessian[r,c]
            mGrads[r]-=curRowVal*mGrads[c]
            for col in range(len(m)):        #Itearting over each element of row
                Hessian[r,col]-=curRowVal*Hessian[c,col]      #Since our value is normalized i.e its 1 so to convert H[r,c] into 0 we need to subtract H[r,c] with 1* H[r,c] but for other elements it would be H[c,col]
    
    # GAUSS JORDAN ENDDSSS AND NOW WE HAVE PARAMETER CAHNGES I.E DELTA WHICH WE WILLL SUBTRACT FROM OUR ORIGNAL SLOPES AND GET OUR NEW BETTER SLOPES

    b-=bGrad
    for i in range(len(m)):
        m[i]-=mGrads[i]

    # for i in range(len(m)):
    #     for j in range(len(m)):
    #         print(Hessian[i,j],end=" ")
    #     print()

    return m,b











#-------------------MAIN EPOCHS AND ALL---------------------------


m=np.zeros(len(x.columns))
b=0
epochs=10
print(lossFunction(m,b,0,0))

for i in range(epochs):
    newtonModel(m,b)
    if(i%2==0):
        print(f"Epoch {i}: {lossFunction(m,b,0,0 )}")
print(f"Epoch 10: {lossFunction(m,b,0,0)}")

        

#---------------------------------FINDING ACCURACY ACCROSS TRAINING AND TESTING SETS-----------------------------

trainSize=len(xTrain)
trainCorrect=0
lmd=0.5
for i in range(trainSize):
    z=b
    for j in range(len(m)):
        z+=m[j]*xTrain.iloc[i,j]
    e=np.exp(-z)
    hx=1/(1+e)
    predY=0
    if(hx>=lmd):
        predY=1
    # print(f"{predY}   {yTrain.iloc[i]}")
    if(yTrain.iloc[i]==predY):
        trainCorrect+=1

testSize=len(xTest)
testCorrect=0
for i in range(testSize):
    z=b
    for j in range(len(m)):
        z+=m[j]*xTest.iloc[i,j]
    e=np.exp(-z)
    hx=1/(1+e)
    predY=0
    if(hx>=lmd):
        predY=1
    if(yTest.iloc[i]==predY):
        testCorrect+=1



print(f"Train Accuracy: {(trainCorrect/trainSize)*100}")
print(f"Test Accuracy: {(testCorrect/testSize)*100}")


#---------------CONFUSION MATRIX AND RELATED METRICS----------------------------


def confusionMatrix(threshold):
    tp,fp,tn,fn=0,0,0,0
    eps=1e-15

    for i in range(testSize):
        z=b
        for j in range(len(m)):
            z+=m[j]*xTest.iloc[i,j]
        e=np.exp(-z)
        hx=1/(1+e)
        yPred=0
        if(hx>=threshold):
            yPred=1
        yActual=yTest.iloc[i]
        if(yActual==1 and yPred==1):
            tp+=1
        elif(yActual==1 and yPred==0):
            fn+=1
        elif(yActual==0 and yPred==1):
            fp+=1
        else:
            tn+=1

    print(f"Accuracy:- {(tp+tn)/(tp+fp+tn+fn+eps)}")                            #Overall Correctness that's why True Positive and True Negative in numerator
    print(f"Precision:- {tp/(tp+fp+eps)}")                                      #How many True values that we are prediciting are actually true over our all predicted True values
    print(f"Recall:- {tp/(tp+fn+eps)}")                                         #How many True values that we are predicting are actually true over all real True values
    print(f"Specificity:- {tn/(tn+fp+eps)}")                                    #How many False values that we are predicting are actually False over all real False values


for i in range(1,10):                                  #Confusion Matrix accorss various thresholds
    print(f"Threshold {i/10} :-")
    confusionMatrix(i/10)

