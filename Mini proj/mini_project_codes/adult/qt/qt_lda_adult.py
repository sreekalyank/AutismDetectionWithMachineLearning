import pandas as pd
import numpy as np
from scipy.io import arff

# Load ARFF file
data, meta = arff.loadarff("C:\\Users\\91939\\Downloads\\Autism-Adult-Data.arff")

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

# Convert dtype_mapping to list of tuples
dtype_tuples = [(col, dtype_mapping[col]) for col in meta.names()]

# Convert to DataFrame with specified data types
df = pd.DataFrame(data, columns=meta.names())

# Apply the specified data types
df = df.astype(dtype_mapping)

# # Print columns and their types
# print("Columns and their types:")
# print(df.dtypes)
# print('DataFrame:')
# print(df)


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


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Assuming categorical_df contains the one-hot encoded categorical columns
# and non_categorical_df contains the bool and float columns

# One-hot encode the categorical columns
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_data = encoder.fit_transform(categorical_df)
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_df.columns))

# Join encoded categorical columns with bool and float columns
joined_df = pd.concat([non_categorical_df, encoded_df], axis=1)

# Display final DataFrame
# print(joined_df)


from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

# Assuming joined_df contains the joined DataFrame

# Separate features and labels
X = joined_df.iloc[:, :-1]  # Features (all columns except the last one)
y = joined_df.iloc[:, -1]   # Labels (last column)

# accuracy_list={}
# Split the data into training and testing sets
# for i in range(1,101):
    
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
    
# Instantiate LabelEncoder
label_encoder = LabelEncoder()
    
# Encode string labels into numerical values
labels_encoded = label_encoder.fit_transform(y)
    
# Now, you can calculate class counts
class_counts = np.bincount(labels_encoded)
    
# Calculate prior probabilities based on class proportions
prior_probabilities = class_counts / len(labels_encoded)
    
# Fit Gaussian distributions to the prior probabilities
means = np.mean(prior_probabilities, axis=0)  # Calculate mean for each class
variances = np.var(prior_probabilities, axis=0)  # Calculate variance for each class
    
# Define the pipeline
pipeline = Pipeline([
    ('transformer', QuantileTransformer(n_quantiles=84,output_distribution='uniform',subsample=350, random_state=15)),
    ('oversampler', RandomOverSampler(random_state=4)),
    ('classifier', LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.25,priors=prior_probabilities, store_covariance=True, tol=0.00009))
])
    
# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)
    
# Perform 10-fold cross-validation on training data
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
    
# Evaluate the model on the testing data
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy on testing dataset:", accuracy)
# accuracy_list[i]=accuracy