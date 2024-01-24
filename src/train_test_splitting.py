import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC

OVERSAMPLING = False


# Load your initial CSV file into a DataFrame
file_path = 'data/labeled_data.csv'
df = pd.read_csv(file_path)

# Separate features (X) and target variable (y)
X = df.drop('class', axis=1)  # Replace 'target_column' with the actual name of your target column
y = df['class']

# Split the initial data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if OVERSAMPLING:
    # Oversample the training set
    smotenc = SMOTENC(categorical_features=["tweet"] ,random_state=42)
    X_train_resampled, y_train_resampled = smotenc.fit_resample(X_train, y_train)

    # Concatenate the oversampled training set into a new DataFrame
    df_train_resampled = pd.concat([pd.DataFrame(X_train_resampled, columns=X.columns), pd.Series(y_train_resampled, name='class')], axis=1)
else:
    df_train_resampled = pd.concat([pd.DataFrame(X_train, columns=X.columns), pd.Series(y_train, name='class')], axis=1)

# Save the testing set to a new CSV file
df_test = pd.concat([X_test, y_test], axis=1)
df_test.to_csv('data/test_file.csv', index=False)

# Save the oversampled training set to a new CSV file
df_train_resampled.to_csv('data/train_resampled_file.csv', index=False)


class_counts = df['class'].value_counts()

print("Class distribution in the original dataset:")
print(class_counts)
print("Class distribution in the trained dataset after split:")
print(df_train_resampled['class'].value_counts())


