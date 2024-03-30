## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1 : Read the given Data.

STEP 2 : Clean the Data Set using Data Cleaning Process.

STEP 3 : Apply Feature Encoding for the feature in the data set.

STEP 4 : Apply Feature Transformation for the feature in the data set.

STEP 5 : Save the data to the file.

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

## 1. FUNCTION TRANSFORMATION

• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation

## 2. POWER TRANSFORMATION

• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```py
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
df=pd.read_csv('Encoding Data.csv')
df
```

![DS_1](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/3bc0e53f-9f0e-4f3f-a2a0-bc25f5c09db6)



```py
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![DS_2](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/145a627e-7ac9-41c8-a901-a2cf7d59f173)


```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```


![DS_3](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/de5e9b2b-0080-4d52-be8b-97ebc9df346d)


```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```


![DS_4](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/6c49cd1c-eb95-46e0-9737-19d384d4bb13)


```py
on=OneHotEncoder(sparse=False)
df2=df.copy()
en=pd.DataFrame(on.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,en],axis=1)
pd.get_dummies(df2,columns=["nom_0"])
```


![DS_5](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/decb4986-93b1-465b-9eec-a48c95103120)


```py
from category_encoders import BinaryEncoder
fd=pd.read_csv('data.csv')
fd
```


![DS_6](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/166f30c6-3abc-4367-8ccc-38379a3d0bc8)


```py
be=BinaryEncoder()
nd=be.fit_transform(fd['Ord_2'])
fd
```

![DS_7](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/edeceacb-b4b4-4d3c-aa85-70bf57f084d8)


```py
dfb=pd.concat([fd,nd],axis=1)
dfb=fd.copy()
dfb
```


![DS_8](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/3990f5b2-898d-4e43-8dca-a8864400885c)



```py
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=fd.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```


![DS_9](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/dca29d3c-1805-4850-a1d6-f5dfc3eb51b5)


```py
from scipy import stats
import numpy as np
ab=pd.read_csv('Data_to_Transform.csv')
ab
```

![DS_10](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/1985cade-ac3c-4602-9e02-e48ab0a9e79b)


```py
ab.skew()
```

![DS_11](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/2032afa6-868b-4a9f-9d06-fb859a30265a)



```py
np.log(ab['Highly Positive Skew'])
```


![DS_12](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/410d5c36-819d-4607-95bc-18d1d818570f)


```py
np.reciprocal(ab["Moderate Negative Skew"])
```


![DS_13](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/c6996421-f72e-4a2c-811b-fc6241b27be0)



```py
np.sqrt(ab["Highly Negative Skew"])
```


![DS_14](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/d7c38da7-8a5d-4f04-b94e-b9d96bf52132)


```py
np.square(ab["Highly Positive Skew"])
```


![DS_15](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/168bb085-8eee-4702-9d60-4e45a61d2b1f)



```py
ab['Highly Positive Skew_boxcox'], parameters=stats.boxcox(ab['Highly Positive Skew'])
ab
```

![DS_16](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/a5817bfc-0340-4505-8ab7-8488b23924f3)


```py
ab.skew()
```

![DS_17](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/57e313b3-2c03-4cf7-b101-625c4c0ca847)



```py
ab['Moderate Negative Skew_yeojohnson'], parameters=stats.yeojohnson(ab['Moderate Negative Skew'])
ab
```

![DS_18](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/8d7dfb5b-0e48-4216-b84a-0d8056e1f792)


```py
ab.skew()

```


![DS_19](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/fa2b4fd6-be58-4daa-bc4b-a47ee1ff8263)


```py
ab['Highly Negative Skew_yeojohnson'], parameters=stats.yeojohnson(ab['Highly Negative Skew'])
ab
```

![DS_20](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/e7f76d9e-6751-473f-a642-1834830dc7c4)


```py
ab.skew()
```


![DS_21](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/766613fb-3978-43a2-97db-b3430215c02b)


```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
ab["Moderate Negative Skew_1"]=qt.fit_transform(ab[["Moderate Negative Skew"]])
ab
```


![DS_22](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/5099533a-b6df-4899-ad54-97724ddca5ab)


```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(ab["Moderate Negative Skew"],line='45')
plt.show()
```

![DS_23](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/10d1d7c1-3047-47b1-80d1-6b402d8edcf1)


```py
sm.qqplot(np.reciprocal(ab["Moderate Negative Skew"]),line='45')
```


![DS_24](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/83f2ebd3-e8ac-4a0f-bd3e-348d9837b6ec)


```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
ab["Moderate Negative Skew"]=qt.fit_transform(ab[["Moderate Negative Skew"]])
sm.qqplot(ab["Moderate Negative Skew"],line="45")
plt.show()
```

![DS_25](https://github.com/Skanthasishanth/EXNO-3-DS/assets/118298456/c025dcee-1f6a-4ab5-a62e-c905d4d0dc72)

      
# RESULT:

Hence performing Feature Encoding and Transformation process is Successful.
       
