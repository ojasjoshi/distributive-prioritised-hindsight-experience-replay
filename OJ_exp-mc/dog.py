import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

mypath = '' 						#default path

if((len(sys.argv)==2)):
	mypath = str(sys.argv[1])

with open(mypath+'train_rew_backup.pkl', 'rb') as f:
    data = pickle.load(f)
# print(len(data))
plt.plot(range(0,len(data)),data,'.')
plt.savefig(mypath+'train.png')
print("plot saved.")