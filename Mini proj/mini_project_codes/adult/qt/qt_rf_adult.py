import pandas as pd
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Assuming 'data.csv' is your dataset
data = pd.read_csv("C:\\Users\\91939\\Downloads\\csv_result-Autism-Adult-Data.csv")

# Assuming 'Class/ASD Traits' is the column you want to predict
X = data.iloc[:, :-1]  # Exclude the last column (target variable)
y = data.iloc[:, -1]  # Target variable
X = X.iloc[:, :-5]

# Splitting data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputation for numerical columns (replace NaN values with mean)
numeric_imputer = SimpleImputer(strategy='mean')
numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[numeric_columns] = numeric_imputer.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = numeric_imputer.transform(X_test[numeric_columns])

# One-hot encoding for categorical columns
categorical_columns = X_train.select_dtypes(include=['object']).columns
onehot_encoder = OneHotEncoder(drop='first', sparse=False)
X_train_encoded = pd.DataFrame(onehot_encoder.fit_transform(X_train[categorical_columns]))
X_test_encoded = pd.DataFrame(onehot_encoder.transform(X_test[categorical_columns]))

# Concatenating encoded features with original dataset
X_train_encoded.index = X_train.index
X_test_encoded.index = X_test.index
X_train = pd.concat([X_train.drop(columns=categorical_columns), X_train_encoded], axis=1)
X_test = pd.concat([X_test.drop(columns=categorical_columns), X_test_encoded], axis=1)

# Convert feature names to string
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Scaling features using Quantile Transformer
scaler = QuantileTransformer()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Classifier
random_forest_classifier = RandomForestClassifier()

# 10-fold Cross Validation
cv_scores = cross_val_score(random_forest_classifier, X_train_scaled, y_train, cv=10)

# Training
random_forest_classifier.fit(X_train_scaled, y_train)

# Testing
y_pred = random_forest_classifier.predict(X_test_scaled)

# Calculating Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy Value:", accuracy)
