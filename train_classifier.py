import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode labels to numerical values (if not already done)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# Initialize the RandomForestClassifier
model1 = RandomForestClassifier(class_weight='balanced')  # Use class_weight='balanced' for imbalanced classes
model2 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
model3= LogisticRegression()
# Train the model
model1.fit(x_train, y_train)

# Predict on the test set
y_predict1 = model1.predict(x_test)

# Evaluate accuracy
score1 = accuracy_score(y_predict1, y_test)
print('{}% of samples were classified correctly by Random Forest Classifier!\n'.format(score1 * 100))


model2.fit(x_train, y_train)

# Predict on the test set
y_predict2 = model2.predict(x_test)

# Evaluate accuracy
score2 = accuracy_score(y_predict2, y_test)
print('{}% of samples were classified correctly by Multilayer Perceptron!\n'.format(score2 * 100))

model3.fit(x_train, y_train)

# Predict on the test set
y_predict3 = model3.predict(x_test)

# Evaluate accuracy
score3 = accuracy_score(y_predict3, y_test)
print('{}% of samples were classified correctly by Random Forest Classifier!'.format(score3 * 100))


# cm = confusion_matrix(y_test, y_predict)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model3, 'label_encoder': label_encoder}, f)

print("Model and label encoder saved successfully.")
