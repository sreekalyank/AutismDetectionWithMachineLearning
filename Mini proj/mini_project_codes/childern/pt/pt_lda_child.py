import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, PowerTransformer  # Replaced Normalizer with PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Import Linear Discriminant Analysis
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, cohen_kappa_score, log_loss, matthews_corrcoef
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("C:\\Users\\91939\\Downloads\\csv_result-Autism-Child-Data.csv")
df.drop(df.columns[0], axis=1, inplace=True)

# Separate features and labels
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

# Select categorical features
category_features = features.iloc[:, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]

# Drop categorical columns
features.drop(features.columns[-10:], axis=1, inplace=True)

# Fill missing values with mean
features = features.fillna(features.mean())

# One Hot Encoding without changing column names
enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
category_transformed = enc.fit_transform(category_features)
category_encoded_columns = enc.get_feature_names_out(category_features.columns)
transformed_df = pd.DataFrame(category_transformed, columns=category_encoded_columns)

# Concatenate encoded features with numerical features
result_df = pd.concat([features, transformed_df], axis=1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_df, labels, test_size=0.85, random_state=38)

# Define Linear Discriminant Analysis classifier
lda = LinearDiscriminantAnalysis()  # Instantiate Linear Discriminant Analysis

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', PowerTransformer())  # Use PowerTransformer instead of Normalizer
        ]), ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), category_encoded_columns)
    ],
    remainder='passthrough'
)

# Create pipeline with preprocessing and LDA classifier for training and testing
pipeline_lda_pt = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lda)  # Use LDA classifier
])

# Define 10-fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation on training set
cv_scores_lda = cross_val_score(pipeline_lda_pt, X_train, y_train, cv=cv)

# Print cross-validation scores
print("Cross-validation scores (LDA with PowerTransformer):", cv_scores_lda)
print("Mean CV accuracy (LDA with PowerTransformer):", np.mean(cv_scores_lda))

# Train the model on the entire training set
pipeline_lda_pt.fit(X_train, y_train)

# Test the model on the separate testing set
y_pred_test_lda = pipeline_lda_pt.predict(X_test)
test_accuracy_test_lda = accuracy_score(y_test, y_pred_test_lda)
print("Testing Set Accuracy with cross-validation (LDA with PowerTransformer):", test_accuracy_test_lda)
for i in range(5):
    print()
# Print precision, F1 score, ROC AUC, recall, kappa score, log loss, and MCC
print("Precision:", precision_score(y_test, y_pred_test_lda, average='macro'))
print("F1-score:", f1_score(y_test, y_pred_test_lda, average='macro'))

# Test the model on the separate testing set
y_prob_test_lda = pipeline_lda_pt.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, y_prob_test_lda))

print("Recall:", recall_score(y_test, y_pred_test_lda, average='macro'))
print("Kappa Score:", cohen_kappa_score(y_test, y_pred_test_lda))
print("Log Loss:", log_loss(y_test, pipeline_lda_pt.predict_proba(X_test)))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred_test_lda))
