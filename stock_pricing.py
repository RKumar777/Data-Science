
# coding: utf-8

# In[20]:


#######################################################
#Homework 3
#Name:  Rishabh Kumar
#andrew ID: rishabh1
#email: rishabh1@andrew.cmu.edu
#######################################################

#Problem 1

##############################
#prepare data for PCA analysis
##############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV

df = pd.read_csv('SP500_close_price.csv')
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
del df['date']
df = df.fillna(method='ffill')
dflog = df.apply(np.log)
#compute absolute returns on the log transformed data
daily_return = dflog.diff(periods=1).dropna()

## Principal Component Analysis using sklearn for correlations (using StandardScaler)

pca=PCA()
Xstd = StandardScaler().fit_transform(daily_return)
prinp=pca.fit_transform(Xstd)
print(prinp.shape)
eigs=np.linalg.eigvals(pca.get_covariance())
cova=pd.DataFrame()
cova["eigen values"]=eigs
cova['component']=df.columns
cova=cova.sort_values(by='eigen values',ascending=False)
totvar=sum(cova['eigen values'])

# Part A
# Plot Eigen Values vs Principal Component

print('\n1.1')
plt.plot(cova['eigen values'],np.arange(cova.shape[0]) + 1,'-ro',linewidth=1)
fig = plt.gcf()
fig.set_size_inches(9, 6)
plt.title('Eigen values vs Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Eigen Value')
plt.savefig('F1.png')
plt.show()


# Plot Cummulative Variance vs Principal Component

print('\n1.2')
cova['cummulative']=np.cumsum(cova['eigen values'])
cova['% variance cumm']=(cova['cummulative']/totvar)*100
plt.plot(np.arange(cova.shape[0])+1,cova['% variance cumm'],linewidth=3)
plt.title('Cummulative Variance vs Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Cummulative Variance %')
fig = plt.gcf()
fig.set_size_inches(9, 6)
plt.savefig('F2.png')
plt.show()



retained80=cova[cova['% variance cumm']<80].shape[0]+1
print('\n1.3')
print('\nIn order to capture atleast 80% of the variance, {} principal components must be retained'.format(retained80))



varret=sum(cova['eigen values'].iloc[0:2])
recerror=(1.0-(varret/totvar))*100
print('\n1.4')
print('\nThe reconstruction error after keeping the first 2 principal components is: {:0.2f} %'.format(recerror))

#Problem 2

firstcomp=cova['component'][0]
plt.plot(daily_return.index,daily_return[firstcomp])
fig = plt.gcf()
fig.set_size_inches(12, 6)
plt.xlabel('Time series')
plt.ylabel('PC1')
plt.title('PC1 vs Time series')
plt.savefig('F3.png')
plt.show()

lowp=min(prinp[:,0:1])
print('The lowest value was {}'.format(lowp))
print('\nThe date on which the lowest value occured was {} which happened due to astock market crash in this year'      .format(daily_return.index[(np.argwhere(prinp==lowp)[0,0])]))


iden=np.identity(cova.shape[0])
wts=pca.transform(iden)
# wts=np.transpose(wts)
wat=pd.DataFrame()
wat['ticker']=cova['component']
wat['PC-1 weight']=wts[:,0]
wat['PC-2 weight']=wts[:,1]
print('\nweights for the first two components are: \n\n{}'.format(wat))

df2=np.genfromtxt('SP500_ticker.csv',dtype=None,delimiter=',',encoding=None,skip_header=1)
df2=pd.DataFrame(df2)
df2.columns=['ticker','company_name','sector']
# print(df2)

water=pd.DataFrame()
water=wat.merge(df2)

watgr=water.groupby(by='sector')['PC-1 weight'].mean()

watgr.plot.bar()
fig = plt.gcf()
fig.set_size_inches(9, 6)
plt.title('Weights(PC-1) vs Sector')
plt.xlabel('Sector')
plt.ylabel('Weights(PC-1)')
plt.savefig('F4.png')
plt.show()

## The graph here shows that Financials are negatively correlated to PC-1 like other sectors but has the greatest effect

watgr2=water.groupby(by='sector')['PC-2 weight'].mean()

watgr2.plot.bar()
fig = plt.gcf()
fig.set_size_inches(9, 6)
plt.title('Weights(PC-2) vs Sector')
plt.xlabel('Sector')
plt.ylabel('Weights(PC-2)')
plt.savefig('F5.png')
plt.show()

## The graph here shows that all the sectors are correlated positively as well as negatively to PC-2 with Utilities
## having maximum effect

## I would use 1st principal component to see the overall market tendencies since it captures
## maximum variance for this dataset hence taking into account all the nuances in the market trends


# Question 2

df3=pd.read_csv('BMI.csv')

def filterfeaturesbycor(daft):
    m=daft.shape[1]
    X=daft.iloc[:,:m-1]
    y=daft.iloc[:,-1]
    jab=pd.DataFrame()
    jab['Features']=daft.columns[0:-1]
    jab['Correlation coefficient']=sklearn.feature_selection.mutual_info_regression(X,y)
    jab=jab.sort_values(by='Correlation coefficient',ascending=False)
    return jab
featar=filterfeaturesbycor(df3)
print('\n2.2')
print('\nThe top three features are: {}, {} and {}'.format(featar['Features'].iloc[0],featar['Features'].iloc[1]                                                     ,featar['Features'].iloc[2]))
print('\n2.3')
def select(daft,d):
    m=daft.shape[1]
    X=daft.iloc[:,:m-1]
    y=daft.iloc[:,-1]
    X_new = SelectKBest(k=d,score_func=f_regression).fit(X, y)
    mask=X_new.get_support()
    maskl=[i for i in range(len(mask)) if mask[i]==True]
    feats=[daft.columns[i] for i in maskl]
    return feats
print('\nFor k=1, the features selected are: {}'.format(select(df3,1)))
print('\nFor k=2, the features selected are: {}'.format(select(df3,2)))
print('\nFor k=3, the features selected are: {}'.format(select(df3,3)))

## Exhaustive subset search is to exhaustively evaluate all possible combinations and select the best possible combination
## This is computationally expensive whereas SelectKBest looks at all possible combinations of K features and selects best
## thus it is computationally less expensive.

print('\n2.4')
m=df3.shape[1]
Xs=df3.iloc[:,:m-1]
ys=df3.iloc[:,-1]
estimator=LinearRegression()
selector = RFECV(estimator)
selector = selector.fit(Xs, ys)
print('\nOptimum number of features for the best model is: {}'      .format(sum([1 if i==True else 0 for i in (selector.support_)])))

