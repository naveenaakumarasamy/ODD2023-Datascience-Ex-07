# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

## CODE
```
Developed by: Naveenaa A K
Reg no : 212222230094
```
### DATA PREPROCESSING BEFORE FEATURE SELECTION:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/a41dc549-40cf-4d23-a1f6-92f665dbd7df)

### CHECKING NULL VALUES:
```
df.isnull().sum()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/a865ae1d-3c08-4780-9cac-069914243d25)

### DROPPING UNWANTED DATAS:
```
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/e091dbfd-0453-48c5-bac0-01a3eb2a9f82)

### DATA CLEANING:
```
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/995a70e6-7144-4a42-a128-066ed9517bcf)

### REMOVING OUTLIERS:
#### Before
```
plt.title("Dataset with outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/9a6c7500-3199-47f9-b40e-9777655e26b4)

#### After
```
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/7141f1d8-54e9-49ef-afe9-37ef8cde2add)

### FEATURE SELECTION:
```
from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/7f48bddc-d49e-4718-a5dc-5933f9cf60e2)

```
from sklearn.preprocessing import OrdinalEncoder
gender = ['male','female']
en= OrdinalEncoder(categories = [gender])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/72c8cde9-a357-449b-be54-d6e19f50dd7f)
```
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/ce46ae72-72d1-4250-b72e-2a1485695ed8)
```
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/ee4cc889-0a24-4add-94a3-e27a027134bd)
```
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"]
```
### FILTER METHOD:
```
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/4a6296fb-f6e9-4582-81ab-7db4f8c76bc7)

### HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
```
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/6dcda4d9-f827-46d7-8348-e1a0b164256b)
### BACKWARD ELIMINATION:
```
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/d4d95fcc-221f-4b17-a033-1bb69d79363f)
### OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
```
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/5f9f1bae-a39b-4f93-9332-4cff9a367161)
### FINAL SET OF FEATURE:
```
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/cec05724-b611-49b3-bd46-1c0c48c96807)

### EMBEDDED METHOD:
```
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
![image](https://github.com/naveenaakumarasamy/ODD2023-Datascience-Ex-07/assets/113497406/675be0fe-dc60-4a93-bb81-778e77797108)

##  RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
