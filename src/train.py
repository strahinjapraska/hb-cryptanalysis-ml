import numpy as np
import pickle 
from sklearn.tree import DecisionTreeClassifier
from lpn_oracle import *


def train_model(n,p,samples): 
    s = np.random.randint(0,2,n)
    lpn = LPNOracle(s,p)

    dt = DecisionTreeClassifier(criterion='entropy')
    A,b = lpn.sample(samples)
    dt.fit(A,b)

    save_model(dt,n,samples)
    save_sampled_data(A,b,s)
    
def save_model(dt,n,samples): 
    filename = 'decision_tree_'+str(n)+'_'+str(samples)+'.joblib'

    with open(filename, 'wb') as file:
        pickle.dump(dt, file)

def save_sampled_data(A,b,s):
    np.savez('sampled_data.npz',A=A,b=b,s=s)


if __name__ == "__main__":
    p = 0.125 
    samples = 10000000
    n = 25

    train_model(n,p,samples)

    