import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

# Load the dataset
file_path = './datasets/banana_quality.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Calculate Z-scores
z_scores = stats.zscore(data.drop('Quality', axis=1))
outliers = (np.abs(z_scores) > 3).any(axis=1)
print(data[outliers])

data_selected = data[~outliers]
numerical_cols = data_selected.drop('Quality', axis=1).columns.values

le = LabelEncoder()

X = data_selected.drop('Quality', axis=1)
y = le.fit_transform(data_selected['Quality'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
logreg_predictions = logreg_model.predict(X_test)

# Create a Support Vector Classifier model
svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predictions = svc_model.predict(X_test)

# Create a Random Forest Classifier model
rfc_model = RandomForestClassifier()
rfc_model.fit(X_train, y_train)
rfc_predictions = rfc_model.predict(X_test)

# Calculate metrics for Logistic Regression
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
logreg_precision = precision_score(y_test, logreg_predictions)
logreg_recall = recall_score(y_test, logreg_predictions)
logreg_f1 = f1_score(y_test, logreg_predictions)

# Calculate metrics for Support Vector Classifier
svc_accuracy = accuracy_score(y_test, svc_predictions)
svc_precision = precision_score(y_test, svc_predictions)
svc_recall = recall_score(y_test, svc_predictions)
svc_f1 = f1_score(y_test, svc_predictions)

# Calculate metrics for Random Forest Classifier
rfc_accuracy = accuracy_score(y_test, rfc_predictions)
rfc_precision = precision_score(y_test, rfc_predictions)
rfc_recall = recall_score(y_test, rfc_predictions)
rfc_f1 = f1_score(y_test, rfc_predictions)

quality_counts = data['Quality'].value_counts()
print(quality_counts)

# Print the metrics
print("Logistic Regression: Accuracy = %.3f, Precision = %.3f, Recall = %.3f, F1 = %.3f" % (logreg_accuracy, logreg_precision, logreg_recall, logreg_f1))
print("Support Vector Classifier: Accuracy = %.3f, Precision = %.3f, Recall = %.3f, F1 = %.3f" % (svc_accuracy, svc_precision, svc_recall, svc_f1))
print("Random Forest Classifier: Accuracy = %.3f, Precision = %.3f, Recall = %.3f, F1 = %.3f" % (rfc_accuracy, rfc_precision, rfc_recall, rfc_f1))