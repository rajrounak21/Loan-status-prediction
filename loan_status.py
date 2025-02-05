import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
loan_dataset = pd.read_csv("loan_data.csv")
print(loan_dataset.head())
print(loan_dataset.shape)
# Remove the rows with any null values
loan_dataset = loan_dataset.dropna()
print(loan_dataset.isnull().sum())
print(loan_dataset.shape)
print(loan_dataset.keys())

# Replace label values  'N' with 0 and 'Y' with 1
loan_dataset.replace({'Loan_Status': {'N': 0, 'Y': 1}}, inplace=True)
print(loan_dataset)
# value counts of Dependent column
value=loan_dataset['Dependents'].value_counts()
print(value)
#replace 3+ to  4
loan_dataset['Dependents'] = loan_dataset['Dependents'].replace(to_replace='3+', value=4)
#again check the values of Dependent column
print(loan_dataset['Dependents'].value_counts())
#replace  in married column No with 0 and Yes with 1
loan_dataset.replace({'Married': {'No': 0,'Yes': 1}},inplace=True)
# replace in Gender column Male with 1 and Female with 0
loan_dataset.replace({'Gender': {'Male': 1,'Female': 0}},inplace=True)
# replace in Education column Graduate with 1 and Not Graduate with 0
loan_dataset.replace({'Education': {'Graduate': 1,'Not Graduate': 0}},inplace=True)
# replace in Self_Employed column Yes with 1 and No with 0
loan_dataset.replace({'Self_Employed': {'Yes': 1,'No': 0}},inplace=True)
#check the values of Property_Area column
value_1=loan_dataset['Property_Area'].value_counts()
print(value_1)

#replace in Property_Area columns Rural:0 , Semiurban:1 , Urban:2
loan_dataset.replace({'Property_Area':{'Rural': 0,'Semiurban': 1,'Urban':2}},inplace=True)
print(loan_dataset.head())
loan_dataset.drop(columns="Loan_ID",axis=1,inplace=True)
loan_dataset.to_csv("clean_loan.csv")
# seperating the features and label
X=loan_dataset.drop(columns=['Loan_Status'],axis=1)
print(X.head())
Y=loan_dataset['Loan_Status']
# split the data into train test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
model=SVC(kernel='linear')
# training the model
model.fit(X_train,Y_train)
#prediction on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy on training data :",training_data_accuracy)

#prediction on testing data
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy on testing data :",testing_data_accuracy)

# testing the model
input_data=(1,	1	,2,	1,	1,	3316,	3500,	88	,360	,1,2)

input_data_as_np_array=np.asarray(input_data)
input_data_reshaped=input_data_as_np_array.reshape(1,-1)
prediction =model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print('The loan is not Granted')

else:
    print("The loan is Granted")
import joblib
joblib.dump(model,"loan.pkl")