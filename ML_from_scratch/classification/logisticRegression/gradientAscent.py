import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split


#-------------------------DATA PREPROCESSING-----------------------------


data=pd.read_csv("breast-cancer.csv")
    
for i,val  in data["diagnosis"].items():
    if val=='M':
        data.at[i,"diagnosis"]=1
    else:
        data.at[i,"diagnosis"]=0

data["diagnosis"]=data["diagnosis"].astype(int)



data=data.drop("id",axis=1) #drop id column as it doesn't provide any meaningful value 
# print(data.head())
# print(data.describe())

# print(data.head())

# sns.pairplot(data)
# plt.show()


#------------------------Splitting features into X and output into Y, and Splitting the data for Training and Testing---------------------------


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

# print(xTrain.head())
# print(yTrain.head())

#----------------------LOSS(Maximum likelihood estimation) AND GRADIENT ASCENT-------------------------------------



def lossFunction(m,b,l1,l2,xTrain,yTrain):
    nlhd=0                                          #negative likelihood=0
    for i in range(len(xTrain)):
        z=b
        for j in range(len(xTrain.iloc[i])):
            z+=m[j]*xTrain.iloc[i,j]
        e=np.exp(-z)                                  #e=e^(-z)
        hx=(1/(1+e))                                  #hx = hthetha(x) = 1/(1+e^(-z))
        eps=1e-15
        hx=np.clip(hx,eps,1-eps)
        curY=yTrain.iloc[i]
        nlhd+=(curY*np.log(hx))+(1-curY)*(np.log(1-hx))               #y*log(hx) + (1-y)(log(1-hx))
    
    ridge=0
    lasso=0
    for i in range(len(m)):
        ridge+=m[i]*m[i]
        lasso+=abs(m[i])
    ridge*=l2
    lasso*=l1
    nlhd=(-nlhd+ridge+lasso)/len(xTrain)
    return nlhd




def gradientAscent(m,b,l,l1,l2,xTrain,yTrain):
    addM=np.zeros(len(m))
    addB=0
    for i in range(len(xTrain)):
        z=b
        for j in range(len(m)):
            z+=m[j]*xTrain.iloc[i,j]
        e=np.exp(-z)
        hx=1/(1+e)
        eps=1e-15
        hx=np.clip(hx,eps,1-eps)
        curY=yTrain.iloc[i]
        addB+=(curY-hx)/(len(xTrain))
        for j in range(len(m)):
            addM[j]+=((curY-hx)*xTrain.iloc[i,j])/len(xTrain)

    b+=(l*addB)
    for i in range(len(m)):
        m[i]+=l*(addM[i] - 2*l2*m[i] - l1*(np.sign(m[i])))

    return m,b



#-------------------------------MAIN CODE EPOCH AND ALL---------------------------------------------------





# l=np.array([ 0.38165748,  0.42986029,  0.35865274,  0.39144489,  0.18306811, -0.23215576,
#   0.5317867,   0.70168187, -0.04334494, -0.2449954,   0.76779355, -0.05071346,
#   0.47828911,  0.53867021,  0.10593973, -0.4380238,   0.04177065,  0.27611858,
#  -0.31643508, -0.43785908,  0.61195981,  0.85171411,  0.49643435,  0.55864984,
#   0.45205257,  0.05253606,  0.65582226,  0.61991184,  0.76167933,  0.07771941])
m=np.zeros(len(x.columns))

# for i in range(len(l)):
#     m[i]=l[i]
    # print(m[i])
# b=-0.5527685592044553
b=0
l=0.8
l1=0
l2=0.004

print(lossFunction(m,b,l1,l2,xTrain,yTrain))

epoch=5000

for i in range(1,epoch+1):
    m,b=gradientAscent(m,b,l,l1,l2,xTrain,yTrain)
    if i%10==0:
        print(f"Epoch {i} :- ", lossFunction(m,b,l1,l2,xTrain,yTrain))
    if i%500==0:
        print(m)
        print(b)





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


