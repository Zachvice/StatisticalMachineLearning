
# coding: utf-8

# In[1]:


from Precode import *
import numpy as np
data = np.load('AllSamples.npy')


# In[2]:


k1,i_point1,k2,i_point2 = initial_S1('7619') # please replace 0111 with your last four digit of your ID


# In[3]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import copy
plt.style.use('ggplot')
colors = [ 'greenyellow', 'slateblue', 'teal', 'orchid', 'indianred','red', 'green', 'yellow', 'blue', 'pink', 'gray', 'brown', 'orange', 'purple',]
class KMeansStrategy1(object):
    def __init__(self, k: int, points: list, data: list):
        self.k = k
        self.points = points
        self.data = data
        self.clusters = None
        self.loss = None
    
    #k: the number of clusters
    #points: 2d list of randomly generated mu values
    #data: 2d list of data points to be classified
    def calculateKMeans(self):
        changed = True
        while changed:
            self.clusters = {}
            for i in range(1, self.k + 1):
                self.clusters[i] = []
            for v in data:
                distance = float('inf')
                curr_cluster = 0
                for i in range(len(self.points)):
                    dist = np.linalg.norm(self.points[i] - v)
                    if dist < distance:
                        distance = dist
                        curr_cluster = i + 1
                self.clusters[curr_cluster].append(v.tolist())
            #Now shift mu of each cluster
            for i in range(len(self.points)):
                mu = np.mean(np.array(self.clusters[i+1]), axis=0)
                if np.array_equal(mu,self.points[i]):
                    changed = False
                else:
                    changed = True
                    self.points[i] = mu
                
    
            
    def calculateObjFunction(self):
        summ = 0
        for i in range(self.k):
            for j in range(len(self.clusters[i+1])):
                summ += np.linalg.norm(self.clusters[i+1][j] - self.points[i]) ** 2
                
        self.loss = summ
    
    def showPlot(self):
        for k in KMS.clusters:
            for point in KMS.clusters[k]:
                plt.plot(point[0],point[1], 'o', color=colors[k-1], label="Cluster='{0}'".format(k) )
            plt.plot(KMS.points[k-1][0],KMS.points[k-1][1], 'o', markersize=15, markeredgewidth=2.0, mec= 'k', color=colors[k-1], label="Cluster='{0}'".format(k) )
        plt.show()
        
    def showInfo(self):
        print('After KMeans Algorithm\n', pd.DataFrame(KMS.points, columns=["X1", "X2"]), '\n')
        print('loss: ', KMS.loss)
            
        
class Helper(object):
    def getRandomPoints(k: int, data: list):
        indices =np.random.choice(data.shape[0], k, replace=False)
        return data[indices]
                
                


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
#For the given points, k's
colors = [ 'greenyellow', 'slateblue', 'teal', 'orchid', 'indianred','red', 'green', 'yellow', 'blue', 'pink', 'gray', 'brown', 'orange', 'purple',]
KMS = KMeansStrategy1(k1, copy.copy(i_point1), data)
KMS.calculateKMeans()
KMS.calculateObjFunction()
KMS.showInfo()
KMS.showPlot()

KMS = KMeansStrategy1(k2, i_point2, data)
KMS.calculateKMeans()
KMS.calculateObjFunction()
KMS.showInfo()
KMS.showPlot()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
#Lets try K clusters from 2 to 10
kms_loss = []
colors = [ 'greenyellow', 'slateblue', 'teal', 'orchid', 'indianred','red', 'green', 'yellow', 'blue', 'pink', 'gray', 'brown', 'orange', 'purple',]
for k in range(2, 11):
    KMS = KMeansStrategy1(k, Helper.getRandomPoints(k, data), data)
    KMS.calculateKMeans()
    KMS.calculateObjFunction()
    kms_loss.append(KMS.loss)
    #KMS.showPlot()
    #KMS.showInfo()
    plt.plot(k, KMS.loss, 'o', markersize=15, markeredgewidth=2.0, mec= 'k', color='b', label="Cluster='{0}'".format(k) )
    plt.xlabel("Number of Clusters")
    plt.ylabel("Loss Value")
    plt.title("Figure 2")
    
    

            
    

