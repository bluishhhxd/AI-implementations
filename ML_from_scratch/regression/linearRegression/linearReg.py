#y=mx+b
#E=(1/n)*Sum(i->n)(actual-pred)^2


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



#----------------------------------DATA CREATION---------------------------------- 

# data=pd.read_csv("advertising.csv")
# data=data[["TV","Sales"]]
# data.to_csv("advertising.csv",index=False)
# print(data.shape)
# data["TV"]=(data["TV"]-data["TV"].mean())/data["TV"].std()
# data["Sales"]=(data["Sales"]-data["Sales"].mean())/data["Sales"].std()

np.random.seed(42)
n=5000
years_of_experience=np.random.randint(2,21,size=n)
m=(200000-60000)/18
b=60000
salaries=m*years_of_experience+b+np.random.normal(0,100000)


data=pd.DataFrame({'Exp':years_of_experience,"Salary":salaries})
# plt.scatter(data.Exp,data.Salary)
# plt.show()

# data=pd.DataFrame(data)

# data["Exp"]=(data["Exp"]-data["Exp"].mean())/data["Exp"].std()
# data["Salary"]=(data["Salary"]-data["Salary"].mean())/data["Salary"].std()


# df.describe()



#Ideal Values for m and b------------------------------------------------------

def idealValue(m,b,data):
    xMean,yMean=0,0
    for i in range(len(data)):
        xMean+=data.iloc[i,0]
        yMean+=data.iloc[i,1]
    xMean/=len(data)
    yMean/=len(data)

    coVar,var=0,0
    for i in range(len(data)):
        x,y=data.iloc[i,0],data.iloc[i,1]
        coVar+=(x-xMean)*(y-yMean)
        var+=(x-xMean)**2
    
    return coVar/var,yMean-m*xMean      #   m=CoVariance/Variance        b=yMean-m*xMean


#MSE as the Loss Function E=(1/n)Sum(i->n)(Yactual - (mx+b))----------------------------------------

def LossFunction(m,b,points):
    totalErrors=0
    for i in range(len(points)):
        # print(x,y)
        x,y=points.iloc[i,0],points.iloc[i,1]
        yPred=m*x+b
        totalErrors+=(y-yPred)**2

    return totalErrors/len(points)

#Gradient Descent Function to find the direction of m and b-----------------------------------------------

def gradientDescent(m,b,points,l):
    mDirec=bDirec=0
    for i in range(len(points)):
        x,y=points.iloc[i,0],points.iloc[i,1]
        dev=y-(m*x+b)
        mDirec+=(-2/len(points))*x*dev
        bDirec+=(-2/len(points))*dev
    
    # mDirec=(-2/len(points))*mDirec
    # bDirec=(-2/len(points))*bDirec

    m=m-l*mDirec
    b=b-l*bDirec
    return m,b


#Main-------------------------------------------------------------------------------------------------------------- 

m,b=idealValue(m,b,data)
print(LossFunction(m,b,data))
l=0.0001
epochs=100
print(m,b)

for i in range(epochs):
    m,b=gradientDescent(m,b,data,l)
    if(i%10==0):
        print(f"Epoch {i}: Loss= {LossFunction(m,b,data)}")

print(m,b)
print(LossFunction(m,b,data))

plt.scatter(data.Exp,data.Salary,color="black")
plt.plot(data.Exp,[m*x+b for x in data.Exp],color="red")
plt.show()

