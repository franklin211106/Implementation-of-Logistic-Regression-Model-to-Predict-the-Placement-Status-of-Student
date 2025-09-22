# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset using pandas and preview the first few rows.

2.Drop irrelevant columns like sl_no and salary.

3.Check and handle missing values or duplicates.

4.Encode categorical variables using LabelEncoder.

5.Define the input features X and target variable y.

6.Split the dataset into training and testing sets using train_test_split.

7.Train a LogisticRegression model on the training data.

8.Make predictions on the test data.

9.Evaluate model performance using:

         Accuracy Score 
         
         Confusion Matrix
         
         Classification Report

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: franklin.f
RegisterNumber:  212224240041
*/
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()


data.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"]) # Encode 'hsc_s'
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x


y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy =accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

### data head

<img width="1593" height="238" alt="image" src="https://github.com/user-attachments/assets/885dbd66-6f1c-45ce-be2d-318f9cd0a1fe" />

### data1 head

<img width="1422" height="241" alt="image" src="https://github.com/user-attachments/assets/548274bb-2154-478f-ba9c-8dd7d62a79a0" />

### isnull

<img width="756" height="638" alt="image" src="https://github.com/user-attachments/assets/5cb06a1b-4def-4752-9768-0f39f5dacc58" />

### duplicate  

<img width="1068" height="64" alt="image" src="https://github.com/user-attachments/assets/823ecff9-a244-4404-a8ab-71ba1b79ac8b" />

### data1  

<img width="1483" height="485" alt="image" src="https://github.com/user-attachments/assets/af708d90-a470-4978-95c4-88f18f884bc2" />

### X  
<img width="1349" height="461" alt="image" src="https://github.com/user-attachments/assets/70941a5b-e46a-4f84-af13-9ba05562aa96" />

### Y  
<img width="972" height="489" alt="image" src="https://github.com/user-attachments/assets/3e13f7dd-1950-48c1-bb2f-d64c23530db8" />

### y_pred 

<img width="944" height="87" alt="image" src="https://github.com/user-attachments/assets/11066ff8-01c7-45dd-9786-054e538bbf89" />

### accuracy

<img width="834" height="78" alt="image" src="https://github.com/user-attachments/assets/4b28553a-6ab6-4ca4-8a1d-64ab005d8842" />

### confusion

<img width="670" height="98" alt="image" src="https://github.com/user-attachments/assets/d553d987-4795-42af-9150-6addc74a3db1" />

### classification

<img width="1008" height="214" alt="image" src="https://github.com/user-attachments/assets/4fcb8c26-ebf0-4385-b85a-cf21312c5242" />

### lr

<img width="1603" height="155" alt="image" src="https://github.com/user-attachments/assets/854246b4-5b83-465d-95c9-0ac5a949ab6b" />





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
