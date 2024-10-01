#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('CAR DETAILS.csv')
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


df.duplicated().sum()


# In[5]:


df.drop_duplicates(inplace=True)
df.duplicated().sum()


# ### Checking Data types

# In[6]:


df.dtypes


# In[7]:


df.shape


# In[8]:


cols=df.columns
cols


# In[9]:


cat_cols=df.dtypes[df.dtypes=='object'].index
num_cols=df.dtypes[df.dtypes!='object'].index
print(cat_cols)
print(num_cols)


# In[10]:


a =  df[num_cols].describe(percentiles=[0.01,0.02,0.03,0.25,0.5,0.75,0.80,0.85,0.90,0.95,0.97,0.98,0.99]).T
a


# In[11]:


df = df[(df['selling_price'] > 0) & (df['km_driven'] > 0)]
print(df.describe())


# In[12]:


print('Year : ',df.year.unique(),'\n')
print('fuel :', df.fuel.unique(), '\n')
print('Transmission : ', df.transmission.unique(), '\n')
print('seller Type: :', df.seller_type.unique(),'\n')
print('Owner :', df.owner.unique(), '\n')


# In[13]:


df['brand']=df['name'].str.split(expand=True)[0]
df['model']=df['name'].str.split(expand=True)[1]
df.head()


# In[14]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()


# In[15]:


df['fuel_type'] = lb.fit_transform(df['fuel'])
df['Type_of_Seller'] = lb.fit_transform(df['seller_type'])
df['Transmit'] = lb.fit_transform(df['transmission'])
df['Owner_Type'] = lb.fit_transform(df['owner'])
df['brand_name'] = lb.fit_transform(df['brand'])
df['model_name'] = lb.fit_transform(df['model'])


# In[16]:


df.head()


# In[17]:


r1=df.groupby('year')['selling_price'].mean()
r1.plot(kind='line',marker='d')
plt.grid()
plt.show()


# In[18]:


r1.plot(kind='bar')
plt.grid()
plt.show()


# In[19]:


r2=df.groupby('brand')['selling_price'].mean()


# In[20]:


r2.plot(kind='bar')
plt.grid()
plt.show()


# In[21]:


cols = ['year','fuel', 'seller_type',
       'transmission', 'owner', 'brand']

plt.figure(figsize=(20,20))
for i in range(len(cols)):
    plt.subplot(3,2,i+1)
    sns.countplot(y=df[cols[i]])
    plt.title(f'Countplot for {cols[i]}')
    plt.grid()

plt.show()


# In[22]:


cols=['year','fuel','seller_type','transmission','owner','brand']
plt.figure(figsize=(30,30))
for i in range(len(cols)):
    plt.subplot(3,2,i+1)
    sns.countplot(x=df[cols[i]],hue=df['brand'])
    plt.title(f'Countplot of {cols[i]} with respect to Selling Price')
    plt.xticks(rotation=45, ha='right')
    plt.grid()
plt.show()


# In[23]:


d1 = df.groupby(df['fuel'])['selling_price'].mean()
d1


# In[24]:


df['selling_price'].mean()


# In[25]:


d2 = d1.isin(d1 > df['selling_price'].mean())
d2


# In[26]:


d3 = df.groupby(df['seller_type'])['selling_price'].mean()
d3


# In[27]:


d4 = d1.isin(d3>df['selling_price'].mean())
d4


# In[28]:


c1 = df.groupby(df['transmission'])['selling_price'].mean()
c1


# In[29]:


c4=c1.isin(c1>df['selling_price'].mean())
c4


# In[59]:


data = df.drop(['name', 'fuel', 'seller_type', 'transmission', 'owner', 'brand','model'], axis=1)
data.head()


# In[60]:


data.to_csv('ds_for_ml.csv')


# In[31]:


data.dtypes


# In[32]:


sns.boxplot(x=data['selling_price'])
plt.title('Boxplot of Selling Price')
plt.show()

sns.boxplot(x=data['km_driven'])
plt.title('Boxplot of Kilometers Driven')
plt.show()

sns.boxplot(x=data['year'])
plt.title('Boxplot of year')

plt.show()


# In[33]:


def treatment_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


# In[34]:


for i in num_cols:
    data = treatment_outliers(data, i)


# In[35]:


sns.boxplot(x=data['selling_price'])
plt.title('Boxplot of Selling Price')
plt.show()

sns.boxplot(x=data['km_driven'])
plt.title('Boxplot of Kilometers Driven')
plt.show()

sns.boxplot(x=data['year'])
plt.title('Boxplot of year')
plt.show()


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


x = data.drop(columns=['selling_price'])
y = data['selling_price']


# In[38]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[39]:


from sklearn.metrics import *


# In[40]:


def eval_model(model,mname):
    model.fit(x_train,y_train)
    ypred = model.predict(x_test)
    train_r2 = model.score(x_train,y_train)
    test_r2 = model.score(x_test,y_test)
    mae = mean_absolute_error(y_test,ypred)
    mse = mean_squared_error(y_test,ypred)
    rmse = np.sqrt(mse)
    res = pd.DataFrame({'Train_R2':train_r2,'Test_R2':test_r2,'MAE':mae,
                       'MSE':mse,'RMSE':rmse},index=[mname])
    return res,ypred


# In[41]:


from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from xgboost import XGBRegressor


# In[42]:


lr = LinearRegression()
lr_res,ypred_lr = eval_model(lr,'LinReg')
lr_res


# In[43]:


dt = DecisionTreeRegressor(max_depth=6,min_samples_split=8)
dt_res,ypred_dt = eval_model(dt,'DT_Reg')
dt_res


# In[44]:


knn = KNeighborsRegressor(n_neighbors=11)
knn_res,ypred_knn = eval_model(knn,'KNN_Reg')
knn_res


# In[45]:


rf = RandomForestRegressor(n_estimators=80,max_depth=6,min_samples_split=8)
rf_res,ypred_rf = eval_model(rf,'RF_Reg')
rf_res


# In[46]:


rg = Ridge()
rg_res,y_pred_rg = eval_model(rg,'Ridge_reg')
rg_res


# In[47]:


ls = Lasso()
ls_res,y_pred_rg = eval_model(ls,'Lasso_reg')
ls_res


# In[48]:


all_res  = pd.concat([lr_res,dt_res,knn_res,rf_res,rg_res,ls_res])
all_res


# In[49]:


import pickle
import joblib


# In[50]:


pickle.dump(rf,open('Best_Model_1.pkl','wb'))
pickle.dump(dt,open('Best_model_2.pkl','wb'))


# In[51]:


load_model=joblib.load('Best_Model_1.pkl')


# In[52]:


random_indices = np.random.choice(data.index, size= 20, replace=False)
sample_data_20 = data.loc[random_indices]
sample_data_20


# In[53]:


sample_data=sample_data_20.drop('selling_price', axis=1)


# In[54]:


Sample_pred = load_model.predict(sample_data)
Sample_pred


# In[55]:


Prediction_sample = pd.DataFrame(Sample_pred)
Prediction_sample


# In[56]:


print(sample_data_20[['selling_price', 'year']])


# In[57]:


sample_data_20['pred_selling_price']=Prediction_sample.values
sample_data_20


# In[58]:


com = (sample_data_20[['selling_price', 'pred_selling_price']])
com.to_csv('sample_prediction.csv')
com


# In[ ]:





# In[ ]:




