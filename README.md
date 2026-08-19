## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="500" height="296" alt="image" src="https://github.com/user-attachments/assets/eeb288a4-d717-4b24-bed3-f81d6e68d1bf" />


# Encoding

 # ORDINAL ENCODER:
 ```
 from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
 pm=['Cold','Warm','Hot']
 e1=OrdinalEncoder(categories=[pm])
 e1.fit_transform(df[["ord_2"]])
 ```
 <img width="727" height="268" alt="image" src="https://github.com/user-attachments/assets/4f19ee8b-d31b-4945-b3a6-68d041ee2876" />

 # LABLE ENCODER:
 ```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['nom_0'])
 dfc
 ```
 <img width="735" height="386" alt="image" src="https://github.com/user-attachments/assets/199d2bbd-7561-4ba2-9ec4-e4d4bb1ec1b6" />

 # ONE HOT ENCODER:
 ```
 from sklearn.preprocessing import OneHotEncoder
 ohe=OneHotEncoder(sparse=False)
 df2=df.copy()
 enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
 df2=pd.concat([df2,enc],axis=1)
 df2
 ```
 <img width="743" height="519" alt="image" src="https://github.com/user-attachments/assets/f094cad8-1313-41bf-ad8d-1b9f6f99b740" />

 # pd.get_dummies(df2,columns=["nom_0"])

 <img width="739" height="340" alt="image" src="https://github.com/user-attachments/assets/cedd5f76-1450-4eed-85ec-700fbb3bfa4a" />


 # BINARY ENCODER:
 ```
 from category_encoders import BinaryEncoder
 import pandas as pd
 df=pd.read_csv("data.csv")
 df
 be=BinaryEncoder()
 nd=be.fit_transform(df['Ord_2'])
 df
 dfb=pd.concat([df,nd],axis=1)
 dfb
 
 ```
 <img width="739" height="468" alt="image" src="https://github.com/user-attachments/assets/b0d41ecb-94e3-44c0-b8fb-1ddb014e53f6" />


 # TARGET ENCODER:
 ```
 from category_encoders import TargetEncoder
 te=TargetEncoder()
 CC=df.copy()
 new=te.fit_transform(X=CC["City"],y=CC["Target"])
 CC=pd.concat([CC,new],axis=1)
 CC
 ```
 <img width="745" height="416" alt="image" src="https://github.com/user-attachments/assets/11127c29-f6c5-4922-9d22-38a753a0b81c" />

# TRANSFORMATION
```
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("Data_to_Transform.csv") 
df
```
<img width="731" height="452" alt="image" src="https://github.com/user-attachments/assets/5cc595ae-b0be-471c-9ebc-3c142b4dfb79" />

# df.skew()
<img width="733" height="141" alt="image" src="https://github.com/user-attachments/assets/75e73d9f-42d4-4559-b12a-7a07ea0f90cc" />

# Log Transformation
```
np.log(df["Highly Positive Skew"])
```
<img width="739" height="257" alt="image" src="https://github.com/user-attachments/assets/4e85f0c3-ecbe-4ead-958a-2218d6bf4b23" />

# Reciprocal Transformation
```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="737" height="255" alt="image" src="https://github.com/user-attachments/assets/cc26e66a-f855-46c3-8e27-2ff5b3205381" />

# Square Root Transformation
```
np.sqrt(df["Highly Positive Skew"])
```
<img width="737" height="253" alt="image" src="https://github.com/user-attachments/assets/015fd206-ea5e-4ffc-a6a8-84551654f268" />

# Square Transformation
```
np.square(df["Highly Positive Skew"])
```
<img width="743" height="259" alt="image" src="https://github.com/user-attachments/assets/5c1d8ed7-8de3-4443-9fce-feea6b91237f" />

# Boxcox method

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="738" height="444" alt="image" src="https://github.com/user-attachments/assets/7967b904-5580-41d9-a964-5bb750e65ce8" />

# Yeojohnson method

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df.skew()
```
<img width="742" height="216" alt="image" src="https://github.com/user-attachments/assets/a4a09b08-2bcf-4379-bf07-3acbbd98d4f1" />

# Quantile Transformation
```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df
```
<img width="746" height="500" alt="image" src="https://github.com/user-attachments/assets/49a26a3b-1e39-4c88-9df2-48c3f765f2da" />

```
import seaborn as sns 
import statsmodels.api as sm 
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="740" height="561" alt="image" src="https://github.com/user-attachments/assets/ff0f62d2-b0b5-4283-85d7-edccff4177f9" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="731" height="500" alt="image" src="https://github.com/user-attachments/assets/efd6f972-bbf7-4eab-8806-e23cb7b08496" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```
<img width="739" height="580" alt="image" src="https://github.com/user-attachments/assets/eaa65873-b712-40a4-87ea-ed07e30fbb33" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]]) 
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
<img width="732" height="503" alt="image" src="https://github.com/user-attachments/assets/b6b5de9c-bf16-4993-af0c-d888ed9f5019" />

```
sm.qqplot(df['Highly Negative Skew_1'],line='45') 
plt.show()
```

<img width="741" height="495" alt="image" src="https://github.com/user-attachments/assets/905bb96f-71bf-4962-83c1-80ee4c90ac51" />

# RESULT:
successfully performed Feature Encoding and Transformation process
       
