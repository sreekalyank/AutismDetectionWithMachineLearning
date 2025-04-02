import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, cohen_kappa_score, log_loss, matthews_corrcoef
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# Load ARFF file
data, meta = arff.loadarff("C:\\Users\\91939\\Downloads\\Autism-Adolescent-Data.arff")

# Define data types mapping
dtype_mapping = {
    'A1_Score': 'bool',
    'A2_Score': 'bool',
    'A3_Score': 'bool',
    'A4_Score': 'bool',
    'A5_Score': 'bool',
    'A6_Score': 'bool',
    'A7_Score': 'bool',
    'A8_Score': 'bool',
    'A9_Score': 'bool',
    'A10_Score': 'bool',
    'age': 'float',
    'gender': 'str',
    'ethnicity': 'str',
    'jundice': 'bool',
    'austim': 'bool',
    'contry_of_res': 'str',
    'used_app_before': 'bool',
    'result': 'float',
    'age_desc': 'str',
    'relation': 'str',
    'Class/ASD': 'str'  # Assuming this is your target variable
}

# Replace missing value symbols ('?' or '') with NaN
for attr in meta.names():
    data[attr] = np.char.strip(np.char.mod('%s', data[attr].astype(str)))
    data[attr][data[attr] == ''] = np.nan

# Convert nominal attributes to strings
for attr in meta.names():
    if meta[attr][0] == 'nominal':
        data[attr] = data[attr].astype(str)

# Convert to DataFrame with specified data types
df = pd.DataFrame(data, columns=meta.names())

# Apply the specified data types
df = df.astype(dtype_mapping)

# Separate columns with nominal values into categorical_df
nominal_columns = [col for col in df.columns if df[col].dtype == 'object']
categorical_df = df[nominal_columns]

# Fill missing values in categorical columns with mode
for col in categorical_df.columns:
    mode_val = categorical_df[col].mode()[0]
    categorical_df[col].fillna(mode_val, inplace=True)

# Separate remaining columns into non_categorical_df
non_categorical_columns = [col for col in df.columns if col not in nominal_columns]
non_categorical_df = df[non_categorical_columns]

# Check for missing values in columns with bool values
bool_columns_with_missing = [col for col in non_categorical_df.columns if non_categorical_df[col].dtype == 'bool' and non_categorical_df[col].isnull().any()]
if bool_columns_with_missing:
    print("Missing values found in columns with bool values. Cannot proceed with mean value imputation.")
else:
    # Apply mean value imputation to float columns in non_categorical_df
    float_columns = [col for col in non_categorical_df.columns if non_categorical_df[col].dtype == 'float64']
    non_categorical_df[float_columns] = non_categorical_df[float_columns].fillna(non_categorical_df[float_columns].mean())

# One-hot encode the categorical columns
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_data = encoder.fit_transform(categorical_df)
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_df.columns))

# Join encoded categorical columns with bool and float columns
joined_df = pd.concat([non_categorical_df, encoded_df], axis=1)

# Separate features and labels
X = joined_df.iloc[:, :-1]  # Features (all columns except the last one)
y = joined_df.iloc[:, -1]   # Labels (last column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# Encode string labels into numerical values
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', Normalizer()),  # Normalizer
    ('oversampler', RandomOverSampler(random_state=4)),  # Oversampling for class imbalance
    ('classifier', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.25, priors=None, store_covariance=True, tol=0.00009))  # LDA classifier
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train_encoded)

# Predict on the testing data
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred)
recall = recall_score(y_test_encoded, y_pred)
roc_auc = roc_auc_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred)
kappa = cohen_kappa_score(y_test_encoded, y_pred)
logloss = log_loss(y_test_encoded, y_pred)
mcc = matthews_corrcoef(y_test_encoded, y_pred)

# Print evaluation metrics
print("\nAccuracy on testing dataset(Normalizer_LDA) :", accuracy)
print("Other Parameters")
print("Precision :", precision)
print("Recall :", recall)
print("ROC AUC :", roc_auc)
print("F1-score :", f1)
print("Kappa :", kappa)
print("Log Loss :", logloss)
print("MCC :", mcc)
