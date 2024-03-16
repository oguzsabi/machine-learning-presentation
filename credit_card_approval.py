import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load the dataset
dataset_path = './datasets/credit_card_approval.csv'
df = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(df.head())

df = df\
    .drop('ZipCode', axis=1)\
    .drop('Ethnicity', axis=1)\
    .drop('Married', axis=1)\
    .drop('Citizen', axis=1)\
    .drop('BankCustomer', axis=1)\

# Identifying categorical columns for one-hot encoding
categorical_cols = ['Gender', 'Industry']

# Numerical columns that will be standardized
numerical_cols = ['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income']

# Remaining columns that do not require preprocessing
remaining_cols = ['PriorDefault', 'Employed', 'DriversLicense', 'Approved']

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
X = df.drop('Approved', axis=1)
y = df['Approved']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# The shape of the transformed data
print(X_train_transformed.shape)
print(X_test_transformed.shape)

# Initialize the model
model = LogisticRegression(random_state=42)

# Fit the model on the training set
model.fit(X_train_transformed, y_train)

# Predict on the test set
y_pred = model.predict(X_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

print(accuracy)
print(conf_matrix)

new_data = {
    'Gender': [1],
    'Age': [28],
    'Debt': [13],
    'Married': [0],
    'BankCustomer': [0],
    'Industry': ['Industrials'],
    'Ethnicity': ['White'],
    'PriorDefault': [0],
    'YearsEmployed': [0],
    'Employed': [0],
    'CreditScore': [1],
    'DriversLicense': [0],
    'Citizen': ['ByBirth'],
    'Income': [150]
}

# Create a DataFrame for the new application
new_data_df = pd.DataFrame(new_data)\
    .drop('Ethnicity', axis=1)\
    .drop('Married', axis=1)\
    .drop('Citizen', axis=1)\
    .drop('BankCustomer', axis=1)\

# Preprocess the new data using the same preprocessor as before
new_data_transformed = preprocessor.transform(new_data_df)

# Use the model to predict the approval outcome
new_application_prediction = model.predict(new_data_transformed)

# Interpret the prediction result
prediction_result = "Approved" if new_application_prediction[0] == 1 else "Not Approved"
print(f"The new application is: {prediction_result}")