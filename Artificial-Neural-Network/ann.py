"""
This Artificial Neural Network uses a bank set to train on and help decide if the
 Customer is going to leave the bank any time soon.
"""
# Artificial Neural Network
# Importing the libraries
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Part 1 - Data Pre processing
data_set = read_csv('Churn_Modelling.csv')  # Importing the dataset
X = data_set.iloc[:, 3:13].values  # need to only include part of the data and not everything included.
y = data_set.iloc[:, 13].values


# Encoding categorical data
label_encoder_X_1 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])  # Replace Strings of Location by Number value

label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])

one_hot_encoder = OneHotEncoder(categorical_features=[1])  # replace label of male female to 0 or 1
X = one_hot_encoder.fit_transform(X).toarray()
X = X[:, 1:]  # Remove the first column which is the dummy variable


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!
artificial_neural_network_classifier = Sequential()  # Initialising the ANN

# Adding the input layer and the first hidden layer
artificial_neural_network_classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
artificial_neural_network_classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
artificial_neural_network_classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
artificial_neural_network_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
artificial_neural_network_classifier.fit(X_train, y_train, batch_size=10, epochs=100)  # Number of Epochs is the
                                                                                       # number of training iterations


# Part 3 - Making predictions and evaluating the model
y_prediction = artificial_neural_network_classifier.predict(X_test)  # Predicting the Test set results
y_prediction = (y_prediction > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_prediction)
print('Model Prediction: \n', y_prediction)
print('Confusion Matrix\n', cm)
accuracy = ((cm[0][0] + cm[0][1])/np.sum(cm))*100.0
print('The accuracy of the model is: ', accuracy)

"""
Customer to predict:
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: 60000
Number of Products: 2
Does the customer have a credit card? yes
Is this customer an active member? Yes
Estimated Salary: 50000
"""
# Part 4 Test Prediction on New Customer
new_customer = np.array(sc.transform([0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]))
new_customer_prediction = artificial_neural_network_classifier.predict(new_customer)
if y_prediction >= 0.5:
    print('Time to let go of this customer.')
else:
    print('Keep this customer by giving him positive feed back.')


