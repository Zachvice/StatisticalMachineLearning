
# coding: utf-8

# In[62]:


import numpy
import scipy.io
import math
import geneNewData



def main():
    myID='7619'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')
    print()
    ave_brightness0, std_dev0, ave_brightness1, std_dev1 = task1_extract_features(train0, train1)
    mean0, var0, mean_f2_0, var_f2_0 = task2_calculate_parameters(ave_brightness0, std_dev0)
    mean1, var1, mean_f2_1, var_f2_1 = task2_calculate_parameters(ave_brightness1, std_dev1)
    print('parameters: ',mean0, var0, mean_f2_0, var_f2_0, mean1, var1, mean_f2_1, var_f2_1)
    posterior, posterior1 = task3_NB_classifier(test0,mean0,var0, mean_f2_0, var_f2_0, test1, mean1, var1, mean_f2_1, var_f2_1)
    task4_scoring(posterior, posterior1, len(test0), len(test1))
    
def task1_extract_features(train0, train1):
    #Extract features from the dataset
    
    brightness0 = []
    std_dev0 = []
    brightness1 = []
    std_dev1 = []
    for img in train0:
        #The average brightness of each image
        brightness0.append(np.average(img))
        #The standard deviation of the brightness of each image
        std_dev0.append(numpy.std(img))
    
    for img in train1:
        #The average brightness of each image
        brightness1.append(np.average(img))
        #The standard deviation of the brightness of each image
        std_dev1.append(numpy.std(img))
    return brightness0, std_dev0, brightness1, std_dev1


def task2_calculate_parameters(ave_brightness, std_dev):
    
    #mean of feature1 for digit
    mean = np.mean(ave_brightness)
    #var of feature 1 for digit
    var = np.var(ave_brightness)
    #mean of feature2 for digit
    mean_f2 = np.mean(std_dev)
    #var of feature2 for digit
    var_f2 = np.var(std_dev)
    
    return mean, var, mean_f2, var_f2 
    
def task3_NB_classifier(test0,mean0,var0, mean_f2_0, var_f2_0, test1, mean1, var1, mean_f2_1, var_f2_1):
    prior_0 = len(test0) / (len(test0) + len(test1))
    brightness0, std_dev0, brightness1, std_dev1 = task1_extract_features(test0, test1)
    posteriors, posteriors2 = [], []
    for x in range(len(brightness0)):
        pdf0 = pdf(brightness0[x], mean0, np.sqrt(var0))
        pdf1 = pdf(std_dev0[x], mean_f2_0, np.sqrt(var_f2_0))
        prob_0 = pdf0 * pdf1 * 0.5
        pdf2 = pdf(brightness0[x], mean1, np.sqrt(var1))
        pdf3 = pdf(std_dev0[x], mean_f2_1, np.sqrt(var_f2_1))
        prob_1 = pdf2 * pdf3 * 0.5
        if prob_0 >= prob_1:
            posteriors.append(0)
        else:
            posteriors.append(1)
        
    for x in range(len(brightness1)):
        pdf0 = pdf(brightness1[x], mean0, np.sqrt(var0))
        pdf1 = pdf(std_dev1[x], mean_f2_0, np.sqrt(var_f2_0))
        prob_0 = pdf0 * pdf1 * 0.5
        pdf2 = pdf(brightness1[x], mean1, np.sqrt(var1))
        pdf3 = pdf(std_dev1[x], mean_f2_1, np.sqrt(var_f2_1))
        prob_1 = pdf2 * pdf3 * 0.5
        if prob_1 >= prob_0:
            posteriors2.append(1)
        else:
            posteriors2.append(0)
        
    return posteriors, posteriors2
            
def task4_scoring(posterior, posterior1, test0_length, test1_length):
    print(posterior)
    count = 0
    for i in posterior:
        if i == 0:
            count += 1
    score0 = count / test0_length
    print(score0)
    count = 0
    for i in posterior1:
        if i == 1:
            count += 1
    score1 = count / test1_length
    print(score1)
    
def pdf(x, mean, std_dev):
    num = np.exp(- ((x - mean ) ** 2) / (2* (std_dev)**2))
    denom = np.sqrt(2*np.pi*(std_dev) ** 2)
    return num / denom
    


if __name__ == '__main__':
    main()

