#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import pickle


# In[2]:


df = pd.read_csv("/Users/AB/OneDrive/Documents/Datasets/Customer_Churn_ECommerce.csv")
df.head()


# In[3]:


df = df.iloc[ :, 1:]
df = df.rename(columns={'PreferredLoginDevice':'LoginDevice', 'WarehouseToHome':"WareDist", 
                        'PreferredPaymentMode':'Payment', 'NumberOfDeviceRegistered':'No_of_devices', 
                       'OrderAmountHikeFromlastYear': 'AmountHike', 'DaySinceLastOrder':'LastOrder',
                       'SatisfactionScore':'SatScore', 'HourSpendOnApp':'AppHours'})
df.head()


# In[4]:


df['Churn'] = df['Churn'].astype(object)
df['CityTier'] = df['CityTier'].astype(object)
df['Complain'] = df['Complain'].astype(object)


# <h2> Handling missing & outlier values </h2>
# 
# Outliers are first detected using Boxplots and are replaced with "NA" values. These values are treated just like missing values and replaced both outliers and genuine missing values with either Mean or Median based on the distribution.
# 
# 

# In[5]:


plt.figure(figsize=(50,30))
sns.boxplot(data=df)
plt.title("Boxplot of all the variables")
plt.xlabel("Variables")
plt.ylabel("Values")


# In[6]:


df.isna().sum()


# In[7]:


def handle_outliers(cols):
    sorted(cols)
    Q1, Q3 = np.percentile(cols, [25,75])
    IQR = Q3 - Q1
    lb = Q1 - (1.5*IQR)
    ub = Q3 + (1.5*IQR)
    return lb, ub


# In[8]:


for cols in df.columns:
    if df[cols].dtype != 'object':
        lr,ur = handle_outliers(df[cols])
        df.loc[df[cols] > ur, cols] = np.nan
        df.loc[df[cols] < lr, cols] = np.nan


# #### Checking the distribution of data in missing & outliers columns to decide whether to replace it with mean or median. 
# #### Most of the columns follow positively skewed distribution, hence replaced them with the median. 
# #### Column "AppHours"  & "No of Devices" follows  normal distribution, replaced their missing & outliers values with the mean. 

# In[9]:


missing_cols = ['Tenure', 'WareDist', 'AppHours', 'AmountHike', 'CouponUsed', 'OrderCount', 'LastOrder', 
                'No_of_devices', 'CashbackAmount', 'NumberOfAddress']


# In[10]:


for col in missing_cols:
    sns.displot(data=df, x=col, kind='kde')


# In[11]:


for col in missing_cols:
    if col != 'AppHours' or 'No_of_devices':
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)
    


# In[12]:


#Segregating categorical and numerical data types from the dataset for efficient analysis and further correlation tests.

cat = []
num = []

for i in df.columns:
    if df[i].dtype == 'object':
        cat.append(i)
    else:
        num.append(i)


# ## Exploratory Data Analysis

# #### Checking if the data is balanced or imbalanced. From the plot, it is clear that the data is imbalanced with more than 80% of the data leaning towards label 0 and only 20% towards 1. 

# In[13]:


sns.countplot(x='Churn', data=df)


# #### Is there any relation between the Gender and customer churn?
# 
# From the below countplot, there seems to be no much difference between the Genders with respect to customers churning. 

# In[14]:


sns.countplot(x='Gender', hue='Churn', data=df)


# #### Is there any evidence that customers with lower sat score are more likely to churn?
# 
# There doesn't seem to be any conclusive evidence that lower satisfaction scores lead to customer churn. There are irregularities in churning with respect to satisfaction score. Highest churn rate is observed at SatScore of 3 while lowest churn rate is observed at SatScore of 2. Similarly, the churn rate at SatScore of 5 is almost same as churn rate at SatScore 3. Hence, there is no strong evidence to condclude that customers with lower SatScore are more likely to churn. 

# In[15]:


churn_satScore_ct = pd.crosstab(index=df['Churn'], columns=df['SatScore'], margins=True, normalize=True)
churn_satScore_ct


# In[16]:


sns.displot(x='SatScore', hue='Churn', col='Churn', data=df)


# In[17]:


for column in cat:
    if column != 'Churn':
        sns.set(style='darkgrid')
        sns.set_palette("hls", 3)
        fig, ax = plt.subplots(figsize=(15,5))
        ax = sns.countplot(x=column, hue='Churn', data=df)

        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 3,
                    '{:1.2f}'.format(height/df.shape[0]),
                    ha='center')
    


# # Feature Selection

# <h3> Correlation between independent variables </h3>
# 
# We consider that there is a strong correlation between two variables if the pearson correlation score is greater than 0.6 or less than -0.6. From the below correlation plot, there seems to be high positive correlation between CouponUsed and OrderCount columns. Hence, we will consider dropping either of these columns to pass into a Machine Learning model in the later stage.

# In[18]:


plt.figure(figsize=(15,10))
corr = df[num].corr()
sns.heatmap(corr, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# <h3> Correlation between Categorical variables </h3>
# 
# #### $H_{0}$: There is no relation between the two categorical variables (OR) Two variables are independent of each other
#     
# #### $H_{a}$: There is a relation between the two categorical variables (OR) Two variables are dependent on each other
# 
# Going by the standard P-value of 0.05 to decide on which Hypothesis to consider. If the P-value is <= 0.05, then we reject the $H_{0}$ that there is no relation and go with the notion that there is a relation between the categorical variables. 
#     
#  

#  - Function to determine the P-value between all categorical variable combinations. 
#  - Since we are only concerned with variables with p-values of <= 0.5, passed an if-statement to print specific p-values.
#  - index != 'Churn' is passed as this is our target or dependent variable and we are only checking for correlation between

# In[19]:


def cat_correlation(index, columns):
    
    try:
        crosstab = pd.crosstab(index=df[index], columns=df[columns])
        (chi2, p, dof, _) = stats.chi2_contingency([crosstab.iloc[0].values, crosstab.iloc[1].values])
        if p <= 0.05 and index != 'Churn':
            message = f'''{index} & {columns} P-Value: {p}'''
            print(message)    
    
    except ValueError:
        return "Some of the values in columns consists 0 leading to ValueError. Hence these are being skipped."
        


# The below nested for-loop will consider all possible combinations of categorical variables and calls the cat_correlation
# function. This will finally result in Categorical variables with p-value <= 0.05. 
# 
# It is evident that LoginDevice, CityTier, and Gender are the most frequent columns with high correlation with other categorical variables. Hence, we will consider dropping these columns before passing the data into ML model.

# In[20]:


for i in range(len(cat)):
    for j in range(i+1, len(cat)):
        cat_correlation(cat[i],cat[j])


# In[21]:


df_model = df.drop(columns=['CouponUsed', 'LoginDevice', 'CityTier', 'Gender'])
df_model.head()


# ### One-Hot Encoding all the categorical variables, so that the parameters can be used in ML model.

# In[22]:


df_encoded = pd.get_dummies(df_model, drop_first=True)
df_encoded.columns


# In[23]:


X = df_encoded.drop(['Churn_1'], axis=1)
y = df_encoded['Churn_1']


# ## Handling the imbalanced nature of dataset using SMOTE resampling technique

# In[24]:


print('Before OverSampling, the shape of X: {}'.format(X.shape)) 
print('Before OverSampling, the shape of y: {} \n'.format(y.shape)) 
  
print("Before OverSampling, counts of label '1': {}".format(sum(y == 1))) 
print("Before OverSampling, counts of label '0': {}".format(sum(y == 0)))


# In[25]:


sm = SMOTE(random_state=33)
X_res, y_res = sm.fit_resample(X, y.ravel())


# In[26]:


print('After OverSampling, the shape of X: {}'.format(X_res.shape)) 
print('After OverSampling, the shape of y: {} \n'.format(y_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_res == 0)))


# In[27]:


X_res = pd.DataFrame(X_res)
y_res = pd.DataFrame(y_res)
y_res.columns = ['Churn_1']

balanced_df = pd.concat([X_res, y_res], axis = 1)


# ## Train-Test Split

# In[28]:


X = balanced_df.drop(['Churn_1'], axis=1)
y = balanced_df['Churn_1']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.70, random_state = 24)


# ## Fitting Logistic Regression Model 

# In[29]:


model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)


# In[30]:


y_train_pred = model_lr.predict(X_train)
y_test_pred = model_lr.predict(X_test)


# In[31]:


train_acc = model_lr.score(X_train, y_train)
test_acc = model_lr.score(X_test, y_test)

print("Training accuracy is: ", train_acc*100)
print("Testing accuracy is :", test_acc*100)


# ## Evaluating Train and Test datasets using Confusion Matrix and Classification Report 

# In[32]:


cm = confusion_matrix(y_train, y_train_pred)
cm
sns.set(font_scale = 1.2)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')


# In[33]:


print(classification_report(y_train, y_train_pred))


# In[34]:


cm = confusion_matrix(y_test, y_test_pred)
cm
sns.set(font_scale = 1.2)
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu")


# In[35]:


print(classification_report(y_test, y_test_pred))


# In[36]:


pickle.dump(model_lr, open('model.pkl', 'wb'))


# In[40]:


X_train.columns


# In[37]:


churn = [
    [8, 15, 3, 2, 3, 4, 13, 2, 3, 169, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    ]


# In[38]:


m = pickle.load(open('model.pkl', 'rb'))
print(m.predict(churn))

