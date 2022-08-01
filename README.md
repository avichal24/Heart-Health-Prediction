
# Heart Disease Prediction

This project uses the Machine Learning algorithm
to predict whether you have a healthy heart or you need to pay
attention towards it.

##  About The Dataset

The dataset comprises of the different values under 
the medical criteria for a normal heart health checkup

Some the sample values are:
1. Chest pain
2. Cholestrol
3. ECG (Electro Cardio Gram)
4. Age etc..

Basically the above values are of medical standards, and moreover
these are the basic required parameters to judge the 
heart health.
In this, it acutally shows the people who had the same values of 
measurement and have suffered from a cronic heart diease, whose treatment
process had started late due to improper prediction from doctors end, and 
our main focus is to solve this issue by precting the heart health prior.

## Project Details

The dataset comprises of the previous patients 
record in which the conclusion for having a heart disease
or not is clearly given in Target column

Now we have splitted the dataset into training and test sets 
in which the accuracy score could be predicted so that we can see
if our model is predicting right or wrong and what is the percentage of 
accuracy of our model.

So, we have applied the Logistic Regression model
in our dataset to predict the best possible result with accuracy of test set around 90%.

![Logistic](https://user-images.githubusercontent.com/109500969/182081520-0b2b9b21-0529-4e9d-8cf7-a569b17eb0b4.jpg)

Finally, to test our model we have entered the input as self
where we analysed both the case scenariaos of two persons,
one having the unhealthy heart and the other with a healthy heart, 
and I am happy to anounce that the model predicted absolutely correct
person with bad heart and a good heart.

## Deployment

To deploy this project run

```bash
### For better understanding use .ipynb format given above

## Importig the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## Defining the Dependent and Independent Variables

df = pd.read_csv('/content/heart_disease_data.csv')
x = df.drop( columns ='target',axis=1).values
y = df['target']

## Spliting into training and test dataset for checking the accuracy

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, stratify=y, test_size=0.2)

### Logistic Regression Model


classifier = LogisticRegression()
classifier.fit(x, y)

### Training set accuracy

train_pred = classifier.predict(x_train)
score = accuracy_score(train_pred.round(), y_train)
print('Accuracy score : ', score)

### Test Set accuracy

test_pred = classifier.predict(x_test)
score_test = accuracy_score(test_pred.round(), y_test)
print('Accuracy Score result for test data is : ', score_test)

## Prediction by Input Data

df.head(3)

input_data_0 = (57,0,1,130,236,0,0,174,0,0,1,1,2)
convert_np = np.asarray(input_data_0).reshape(1,-1)
input_pred = classifier.predict(convert_np)
if(input_pred== 1):
 print('WOW! Congrats Yor Heart is good!!')
else:
 print('You need a detailed checkup')

input_data_1 = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)
convert_np = np.asarray(input_data_1).reshape(1,-1)
input_pred = classifier.predict(convert_np)
if(input_pred == 0):
 print('Congrats Yor Heart is good!!')
else:
 print('Attention: You need a detailed checkup')

```

## Conclusion

Now, eversince the model is trained perfectly 
therefore it can be a great helping hand for the medical staff
to predict if the patient at their door is healthy from heart or 
not, and would surely help the person to get a proper treatment for the 
diagnosis he had gone through my model.

# Hi, I'm Avichal Srivastava ! 👋

You can reach out to me at: srivastavaavichal007@gmail.com 

LinkedIn: www.linkedin.com/in/avichal-srivastava-186865187

## 🚀 About Me

I'm a Mechanical Engineer by education, and love to work with data, I eventually started my coding journey for one of my Drone project, wherein I realized that it is something which makes me feel happy in doing, then I planed ahead to move in the Buiness Analyst or Data Analyst domain. The reason for choosing these domains is because I love maths a lot and all the Machine Learning algorithms are completely based on mathematical intution, So this was about me Hope! You liked it, and it is just a beginning, many more to come, till then Happy Analysing!!

