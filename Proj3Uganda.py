#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Configuration Code
get_ipython().run_line_magic('pip', 'install gspread_pandas')
get_ipython().run_line_magic('pip', 'install fooddatacentral')
get_ipython().run_line_magic('pip', 'install pint')
get_ipython().run_line_magic('pip', 'install cufflinks')
get_ipython().run_line_magic('pip', 'install CFEDemands')


from  scipy.optimize import linprog as lp
import numpy as np
import warnings
import pandas as pd
import eep153_tools
from eep153_tools.sheets import read_sheets
import fooddatacentral as fdc
import cufflinks as cf


from cfe import Regression

cf.go_offline()


# In[4]:


#(A) Choice of Population, with supporting expenditure

print('We chose to analyze the Ugandan popultion of males and females 19-30') 
Uganda_Data = '1yVLriVpo7KGUXvR3hq_n53XpXlD5NmLaH1oOMZyV0gQ'

x = read_sheets(Uganda_Data,sheet='Expenditures (2019-20)') #expenditures of 2019-20 
x.columns.name = 'j'
d = read_sheets(Uganda_Data,sheet="HH Characteristics") #household characteristics 
d.columns.name = 'k'
x = x.groupby('j',axis=1).sum() #reducing duplicate columns
x = x.replace(0,np.nan) #reducing nulls
y = np.log(x.set_index(['i','t','m'])) #log of expenditure 
d.set_index(['i','t','m'],inplace=True) #specific labels for the axis


use = y.index.intersection(d.index)
y = y.loc[use,:]
d = d.loc[use,:]


#Filtering it down to our population of interest (M,F 19-30) 
b = read_sheets(Uganda_Data,sheet='RDI')
b = b.set_index('n')



# In[5]:


#(A) Estimate Demand System
from cfe.estimation import drop_columns_wo_covariance
y = drop_columns_wo_covariance(y,min_obs=30)
use = y.index.intersection(d.index)
y = y.loc[use,:]
d = d.loc[use,:]

#y is log expednitures on food j by household i at a particular time
y = y.stack()
d = d.stack()
assert y.index.names == ['i','t','m','j']
assert d.index.names == ['i','t','m','k']

#setting up the regression
result = Regression(y=y,d=d)
#predicting expenditures
result.predicted_expenditures()

#Compare predicted log expenditures with actual 
get_ipython().run_line_magic('matplotlib', 'notebook')
df = pd.DataFrame({'y':y,'yhat':result.get_predicted_log_expenditures()})
df.plot.scatter(x='yhat',y='y')


# In[6]:


#Demand and Household Composition
result.gamma


# In[7]:


#(B) Nutritional Content of Different Foods


# In[8]:


#(B) Nutritional Adequacy of Diet

#helper function
def helper(age, sex): 
    if (age < 4):
        group = str(sex) + ' 00-03'
    elif age < 9: 
        group = str(sex) + ' 04-08'
    elif age < 14:
        group = str(sex) + ' 09-13'
    elif age < 19:
        group = str(sex) + ' 14-18'
    elif age < 31:
        group = str(sex) + ' 19-30'
    elif age < 51:
        group = str(sex) + ' 31-50'
    else: # over 51: 
        group = str(sex) + ' 51+'
    return group


def dietary_ref_index(age, sex):
    group = helper(age, sex)
    series = (b[group])
    return series 

dietary_ref_index(22, 'M')

#Create a table where the expenditures are mapped to the foods and their correspodning nutritions
#compare the nutritional expenses to the RDI guide 
#where are they lacking, or what percent of households are eating adequatley 



# In[9]:


#(C) Counterfactual Experiments

#maybe do this 


# In[ ]:




