
# coding: utf-8

# In[1]:


# PART A

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
hwithm=pd.read_csv('house_with_missing.csv')
hnom=pd.read_csv('house_no_missing.csv')
colnames=['Attribute_ID','Attribute_Name Missing','Mean', 'Median', 'Sdev', 'Min', 'Max']
rows1=hnom.shape[0]
cols1=hnom.shape[1]
rows2=hwithm.shape[0]
cols2=hwithm.shape[1]
# sxv1=[]
# rxv1=[]
# sxv2=[]
# rxv2=[]
# sxvnames1=[]
# rxvnames1=[]
# sxvnames2=[]
# rxvnames2=[]
            
            
# Function to format the data obtained from describe using functions like: transpose, drop, add, reset_index etc.
def brief(data):
    sv=[]
    rv=[]
    svnames=[]
    rvnames=[]
    missval=[]
    unq=[]
    inds=[]
    gump=[]
    q=0
    jump=[]
    joo=0
    sentence3='real valued attributes'
    sentence4='symbolic attributes'
    rws=data.shape[0]
    clsx=data.shape[1]
    axel=data.describe()
    axel=axel.transpose()
    
    # cutting unrequired data from the dataframe and changing the order of the required fields
    
    axel.drop(labels=["25%","75%"],axis=1)
    axel=axel[['count','mean','50%','std','min','max']]
    axel['count']=rws-axel['count']
    
    # changing column names to required names and adding a few columns as per requirements
    
    axel=axel.rename(index=str,columns={'count':'Missing','50%':'Median','std':'Sdev'})
    for i in range(data.shape[1]):
        if(data.dtypes[i]=='O'):
            sv.append(i+1)
            svnames.append(data.columns.values[i])
        else:
            rv.append(i+1)
            rvnames.append(data.columns.values[i])
    axel.insert(0,'Attribute_ID',rv)
    axel.insert(1,'Attribute_Name',rvnames)
    inds=np.arange(1,len(rv)+1)
    axel=axel.reset_index(drop=True)
    axel.index=(np.arange(1, len(axel) + 1))
    
    print(sentence3)
    print('-'*len(sentence3))
    print(axel)
    axel.to_csv('File1.csv')
    stats.write('\n'+sentence3+'\n')
    stats.write('-'*len(sentence3)+'\n')
    stats.write(str(axel))
    
    dt={'Attribute_ID':sv,'Attribute_Name':svnames}
    vitsel=pd.DataFrame(data=dt)
    for j in svnames:
        missval.append((data[j].isna().sum()))
        if((data[j].isna().sum())>0):
            unq.append(len(data[j].unique())-1)
        else:
            unq.append(len(data[j].unique()))
            
    # trying to create a new dataframe for the symbolic attributes applying similar functions as before      
    vitsel.insert(2,'Missing',missval)
    vitsel.insert(3,'Arity',unq)
    vitsel.insert(4,'MCVs_Count',np.arange(0,vitsel.shape[0],1))
    
    # addition of the MCVs_Count column 
    for j in svnames:
        jump=data[j].unique()
        jump=pd.DataFrame(data=jump,columns=['unq'])
        jump=jump.dropna()
        jump=jump.reset_index(drop=True)
        gump=data[j].value_counts(dropna=True)
        if(len(jump)==1):
            vitsel.iloc[q,4]='{}({})'.format(jump['unq'][0],gump[0])
            q=q+1
        if(len(jump)==2):
            vitsel.iloc[q,4]='{}({}) {}({})'.format(jump['unq'][0],gump[0],jump['unq'][1],gump[1])
            q=q+1
        if(len(jump)==3):
            vitsel.iloc[q,4]='{}({}) {}({}) {}({})'.format(jump['unq'][0],gump[0],jump['unq'][1],gump[1],jump['unq'][2],gump[2])
            q=q+1
    vitsel.index=(np.arange(1, len(vitsel) + 1))
    print('\n'+sentence4)
    print('-'*len(sentence4))
    print('{}'.format(vitsel))
    stats.write('\n \n'+sentence4)
    stats.write('\n'+'-'*len(sentence4))
    stats.write('\n'+'{}'.format(vitsel))
    
# printing with proper format

sentence1='brief function output for house_with_missing.csv'
sentence2='brief function output for house_no_missing.csv'
stats=open("statistics.txt","w")
print('~'*len(sentence1))
print(sentence1)
print('~'*len(sentence1))
print('This dataset has {} Rows {} Attributes\n'.format(rows1,cols1))
stats.write('~'*len(sentence1)+'\n')
stats.write(sentence1+'\n')
stats.write('~'*len(sentence1)+'\n')
stats.write('This dataset has {} Rows {} Attributes\n'.format(rows1,cols1))
stats.write('\n \n')
brief(hnom)

print('\n \n')
print('~'*len(sentence2))
print(sentence2)
print('~'*len(sentence2))
print('This dataset has {} Rows {} Attributes\n'.format(rows2,cols2))

stats.write('\n \n'+'~'*len(sentence2)+'\n')
stats.write(sentence2+'\n')
stats.write('~'*len(sentence2)+'\n')
stats.write('This dataset has {} Rows {} Attributes\n'.format(rows2,cols2))
stats.write('\n \n')
brief(hwithm)
stats.close()
#PART B

mx1=hnom[hnom.columns.values[0]].max()
rnz1=(0,mx1)
bins=50
mx2=hwithm[hwithm.columns.values[0]].max()
rnz2=(0,mx2)
hwithm=hwithm.dropna(axis=0)
hnom=hnom.dropna(axis=0)
# plotting a histogram for file 1
plt.hist(hnom[hnom.columns.values[0]], bins, rnz1, color = 'green', 
        histtype = 'bar', rwidth = 0.8) 
  
# x-axis label 
plt.xlabel('{}'.format(hnom.columns.values[0])) 
# frequency label 
plt.ylabel('Total')
# plot title 
plt.title('Histogram for File 1: {}'.format(hnom.columns.values[0])) 
plt.savefig('Histo_1.png')
# function to show the plot 
plt.show() 

# plotting a histogram for file 2
plt.hist(hwithm[hwithm.columns.values[0]], bins, rnz2, color = 'green', 
        histtype = 'bar', rwidth = 0.8) 
  
# x-axis label 
plt.xlabel('{}'.format(hwithm.columns.values[0])) 
# frequency label 
plt.ylabel('Total')
# plot title 
plt.title('Histogram for File 2: {}'.format(hwithm.columns.values[0])) 
plt.savefig('Histo_2.png')
# function to show the plot 
plt.show() 

# Comparison of 2 variables

plt.hist2d(hnom[hnom.columns.values[3]],hnom[hnom.columns.values[0]])
# x-axis label 
plt.xlabel('{}'.format(hnom.columns.values[3])) 
# frequency label 
plt.ylabel('{}'.format(hnom.columns.values[0]))
# plot title 
plt.title('Histogram for File 1: {} vs {}'.format(hnom.columns.values[3],hnom.columns.values[0]))
plt.savefig('Histo2d_3.png')
plt.show()

#plotting histogram for file2
plt.hist2d(hwithm[hwithm.columns.values[3]],hwithm[hwithm.columns.values[0]])
# x-axis label 
plt.xlabel('{}'.format(hwithm.columns.values[3])) 
# frequency label 
plt.ylabel('{}'.format(hwithm.columns.values[0]))
# plot title 
plt.title('Histogram for File 2: {} vs {}'.format(hwithm.columns.values[3],hwithm.columns.values[0]))
plt.savefig('Histo2d_4.png')
plt.show()


# Comparison of 2 variables

plt.scatter(hnom[hnom.columns.values[1]],hnom[hnom.columns.values[8]])
# x-axis label 
plt.xlabel('{}'.format(hnom.columns.values[1])) 
# frequency label 
plt.ylabel('{}'.format(hnom.columns.values[8]))
# plot title 
plt.title('Histogram for File 1: {} vs {}'.format(hnom.columns.values[1],hnom.columns.values[8]))
# plt.legend()
plt.savefig('Histo2d_5.png')
plt.show()

#plotting histogram for file2
plt.scatter(hwithm[hwithm.columns.values[1]],hwithm[hwithm.columns.values[8]])
# x-axis label 
plt.xlabel('{}'.format(hwithm.columns.values[1])) 
# frequency label 
plt.ylabel('{}'.format(hwithm.columns.values[8]))
# plot title 
plt.title('Histogram for File 2: {} vs {}'.format(hwithm.columns.values[1],hwithm.columns.values[8]))
plt.savefig('Histo2d_6.png')
plt.show()


# In[2]:


import pandas
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model
import matplotlib.pyplot as plt

# Use pandas.read_csv to import the data
hwithm=pd.read_csv('house_with_missing.csv')
hnom=pd.read_csv('house_no_missing.csv')

# Connect-the-dots model that learns from train set and is being tested using test set
# Assumes inputs are pandas data frames
# Assumes the last column of data is the output dimension
def get_pred_dots(train,test):
    n,m = train.shape # number of rows and columns
    X = train.iloc[:,:m-1]# get training input data
    query = test.iloc[:,:m-1]# get test input data
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(X)
    distances, nn_index = nbrs.kneighbors(query)# Get two nearest neighbors
    pred = (train.iloc[nn_index[:,0],m-1].values+train.iloc[nn_index[:,1],m-1].values)/2.0
    return pred


# Linear model
# Assumes the last column of data is the output dimension
def get_pred_lr(train,test):
    pred=[]
    # Your implementation goes here
    # You may leverage the linear_model module from sklearn (scikit-learn)
    n,m=train.shape
    regr=linear_model.LinearRegression()
    X = train.iloc[:,:m-1]# get training input data
    query = train.iloc[:,-1]# get test input data
    regr.fit(X,query)
    pred=regr.predict(test.iloc[:,:m-1])
    
    return pred

# Default predictor model
# Assumes the last column of data is the output dimension
def get_pred_default(train,test):
    pred=[]
    preda=[]
    n,m=train.shape
    g=test.shape[0]
    # Your implementation goes here
    X=train.iloc[:,-1]
    pred=(X.mean())
    preda=np.ones(g)
    preda=preda*pred
    return preda


def do_cv(df,output,k,func):
    n,m=df.shape
    remain=np.remainder(n,k)
    times=int(n/k)
    respre=[]
    resinp=[]
    res=[]
    for j in range(k):
        s1=times*(j)
        s2=times*(j+1)
        dftest=df.iloc[s1:s2]
        dftrain=df.drop(df.index[s1:s2])
        testrws=dftest.shape[0]
        respre=func(dftrain,dftest)
        resinp=dftest.iloc[:,-1]
        res.append((np.sum((np.square(resinp-respre)))/testrws))
#         print((np.sqrt(np.sum((np.square(resinp-respre)))/testrws)))
#         print(resinp)
#         print(respre)
    res1=np.array(res)  
#     print(res1)
    return res1

dumd=hnom['house_value']
hnom=hnom.drop(columns=['house_value','num_of_rooms','Charles_river_bound','dist_to_employment_center','property_tax_rate','student_teacher_ratio','Nitric_Oxides','accessiblity_to_highway'])
lim=hnom.shape[1]
coo=hnom.shape[0]
hnom.insert(lim,'house_value',dumd)
hnom.iloc[:,0]=np.log(hnom.iloc[:,0])
# print('\n \n The values for connect-the-dots model')
# print(do_cv(hnom,'house_value',coo,get_pred_dots))
# print('\n \n The values for default model')
# print(do_cv(hnom,'house_value',coo,get_pred_default))
# print('\n \n The values for linear regression model')
# print(do_cv(hnom,'house_value',coo,get_pred_lr))

#function to calculate the interval values
def confint(maal):
    zee=1.96
    meena=np.mean(maal)
    num=np.shape(maal)[0]
    maal2=np.ones(num)
    maal2=maal2*meena
    sdeva= np.sqrt(np.sum(np.square(np.subtract(maal,maal2)))/num)
    hp=meena+(sdeva/np.sqrt(num))
    hn=meena-(sdeva/np.sqrt(num))
    return[hp,meena,hn]
lrresult=confint(do_cv(hnom,'house_value',coo,get_pred_lr))
dotsresult=confint(do_cv(hnom,'house_value',coo,get_pred_dots))
defresult=confint(do_cv(hnom,'house_value',coo,get_pred_default))
# width of the bars
barWidth = 0.5
 
# Choose the height of the blue bars=mean
bars1 = [lrresult[1],dotsresult[1],defresult[1]]
 
# Choose the height of the error bars (high-low interval)
yer1 = [lrresult[0]-lrresult[2],dotsresult[0]-dotsresult[2],defresult[0]-defresult[2]]
print(dotsresult[0]-dotsresult[2])
# The x position of bars
r1 = np.arange(len(bars1))
 
# Create different bars
plt.bar(r1, bars1, width = barWidth, color =['blue'], edgecolor = 'black', yerr=yer1, capsize=7)
 
# general layout
plt.xticks([r for r in range(len(bars1))], ['Linear_regression', 'Connect_the_dots','Default'])
plt.ylabel('house_value')
plt.legend(['house value'])
plt.title('95% Confidence Interval')
 
# Show graphic
plt.savefig('Conf95.png')
plt.show()
    


# In[3]:


print(hnom)
hnom=hnom.sample(hnom.shape[0])
print(hnom)

