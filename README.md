# Cryptanalysis of HB protocol with Machine Learning
- Core of project is located in notebook, project.ipynb

- To re-run the statistics shown in decision tree part:  
<code>python stats.py</code><br>
...it will generate .csv file in data dir 

- Best tree model generated needs to be trained(too large size) with: <br>
<code>python train.py</code> <br>
...then you can run the cell importing it 

- Neural network is saved in model dir you can load it and predict the key for n = 29, p = 0.125 , samples= 2m. Real secret key for that problem is in secret_key.npz you can load it and check those 2 values
- Packages and libraries that are needed are located in Imports section of notebook 
