import pandas as pd
import os.path
import numpy as np



x_min=-np.pi/2 + 0.1
x_max=np.pi/2 - 0.1
domain_length=x_max-x_min


def function_to_approx(x,x2=0):
    return 3*np.tan(x) +2*x2*x2

def train_data_2d(steps):
    [x1,x2,y] = [[],[],[]]
    for i in range(steps):
        for j in range(steps):
            x1p=x_min+ domain_length*i/steps
            x2p=x_min+ domain_length*j/steps
            x1.append(x1p)
            x2.append(x2p)
            y.append(function_to_approx(x1p,x2p))
    return [x1,x2,y]



def train_data(steps):
    [x,y] = [[],[]]
    for i in range(steps):
        xp=x_min+ domain_length*i/steps
        x.append(xp)
        y.append(function_to_approx(xp))
    return [x,y]

def test_data(steps):
    [x,y] = [[],[]]
    for i in range(steps):
        xp=x_min+np.random.random() *domain_length
        x.append(xp)
        y.append(function_to_approx(xp))
    return [x,y]




def write_train_data(steps):
    if os.path.isfile("../input/train_data.csv") == True:
        print("File already exists.")
    else:
        dset=train_data(steps)
        df = pd.DataFrame(data={"x": dset[0], "y": dset[1]})
        df.to_csv("../input/train_data.csv", sep=',',index=True)


def write_train_data_2d(steps):
    if os.path.isfile("../input/train_data.csv") == True:
        print("File already exists.")
    else:
        dset=train_data_2d(steps)
        print(dset)
        df = pd.DataFrame(data={"x1": dset[0],"x2": dset[1] ,"y": dset[2]})
        df.to_csv("../input/train_data.csv", sep=',',index=False)


def write_test_data(steps):
    if os.path.isfile("../input/test_data.csv") == True:
        print("File already exists.")
    else:
        dset=test_data(steps)
        df = pd.DataFrame(data={"x": dset[0], "y": dset[1]})
        df.to_csv("../input/test_data.csv", sep=',',index=True)


write_train_data_2d(20)