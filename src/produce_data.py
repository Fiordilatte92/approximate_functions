import pandas as pd
import os.path
import numpy as np

x_min=-np.pi/2 + 0.1
x_max=np.pi/2 - 0.1
domain_length=x_max-x_min
def function_to_approx(x):
    return np.tan(x)

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


def write_test_data(steps):
    if os.path.isfile("../input/test_data.csv") == True:
        print("File already exists.")
    else:
        dset=test_data(steps)
        df = pd.DataFrame(data={"x": dset[0], "y": dset[1]})
        df.to_csv("../input/test_data.csv", sep=',',index=True)