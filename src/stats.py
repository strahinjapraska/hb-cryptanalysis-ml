from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd 
from lpn_oracle import * 



if __name__ == "__main__":
    p = 0.125 


    samples = [10,10000,100000,1000000,10000000]
    max_dim = 25 

    it = 1 
    iters = max_dim*len(samples)
    df = pd.DataFrame({'samples':[],'dimension':[],'percentage':[],'success':[]}) 

    for sample in samples: 
        for n in range(1,max_dim+1): 
            s = np.random.randint(0,2,n)
            lpn = LPNOracle(s,p)

            dt = DecisionTreeClassifier(criterion='entropy')
            A,b = lpn.sample(sample)
            dt.fit(A,b)

            s_prime = dt.predict(np.eye(n))
            t = calculate_threshold(n,p,sample)

            if check_prediction(A,b,s_prime,t):
                df.loc[len(df.index)] = [sample, n, round(np.mean(s_prime==s)*100,2), 1]
            
            else:
                df.loc[len(df.index)] = [sample, n, round(np.mean(s_prime==s)*100,2), 0] 
            print(str(it)+"/"+str(iters))
            it+=1

    df.to_csv('data/data.csv',index=False)


