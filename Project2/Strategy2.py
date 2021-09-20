
# coding: utf-8

# In[1]:


from Precode2 import *
import numpy
data = np.load('AllSamples.npy')


# In[2]:


k1,i_point1,k2,i_point2 = initial_S2('7619') # please replace 0111 with your last four digit of your ID


# In[3]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
colors = [ 'greenyellow', 'slateblue', 'teal', 'orchid', 'indianred','red', 'green', 'yellow', 'blue', 'pink', 'gray', 'brown', 'orange', 'purple',]
class KMeansStrategy2(object):
    def __init__(self, k: int, points: list, data: list):
        self.k = k
        self.points = [points]
        self.data = data
        self.clusters = None
        self.loss = None
        
    def _addPoint(self, indices):
        maxx = 0
        for i in range(len(self.data)):
            if i in indices.keys():
                continue
            runningDist = 0
            for j in range(len(self.points)):
                runningDist += np.linalg.norm(abs(self.points[j] - self.data[i]))
            if runningDist > maxx:
                maxx = runningDist
                index = i
        indices[index] = 1
        self.points.append(self.data[index])
        
    def calculateInitialCenters(self):
        indices = {}
        for i in range(self.k - 1):
            self._addPoint(indices)
        
            
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
                    
    def showPlot(self):
        for k in KMS.clusters:
            for point in KMS.clusters[k]:
                plt.plot(point[0],point[1], 'o', color=colors[k-1], label="Cluster='{0}'".format(k) )
            plt.plot(KMS.points[k-1][0],KMS.points[k-1][1], 'o', markersize=15, markeredgewidth=2.0, mec= 'k', color=colors[k-1], label="Cluster='{0}'".format(k) )
        plt.show()
        
    def showInfo(self):
        print('After KMeans Algorithm\n', pd.DataFrame(KMS.points, columns=["X1", "X2"]), '\n')
        print('loss: ', KMS.loss)
            
    
    def calculateObjFunction(self):
        summ = 0
        for i in range(self.k):
            for j in range(len(self.clusters[i+1])):
                summ += np.linalg.norm(np.array(self.clusters[i+1][j]) - np.array(self.points[i])) ** 2
                
        self.loss = summ
        
class Helper(object):
    def getRandomPoint(k: int, data: list):
        indices =np.random.choice(data.shape[0], 1, replace=False)
        return data[indices]
    
    
        


# In[5]:


KMS = KMeansStrategy2(k1, i_point1, data)
KMS.calculateInitialCenters()
KMS.calculateKMeans()
KMS.calculateObjFunction()
KMS.showInfo()
KMS.showPlot()



KMS = KMeansStrategy2(k2, i_point2, data)
KMS.calculateInitialCenters()
KMS.calculateKMeans()
KMS.calculateObjFunction()
KMS.showInfo()
KMS.showPlot()


# In[7]:


for k in range(2, 11):
    KMS = KMeansStrategy2(k, Helper.getRandomPoint(1, data), data)
    KMS.calculateInitialCenters()
    KMS.calculateKMeans()
    KMS.calculateObjFunction()
    KMS.showPlot()
    KMS.showInfo()

