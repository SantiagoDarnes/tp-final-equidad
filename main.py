import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Define column names based on the documentation
columns = [
    'checking_account',      # Status of existing checking account
    'duration',             # Duration in month
    'credit_history',       # Credit history
    'purpose',             # Purpose
    'credit_amount',       # Credit amount
    'savings_account',     # Savings account/bonds
    'employment',          # Present employment since
    'installment_rate',    # Installment rate in percentage of disposable income
    'personal_status',     # Personal status and sex
    'other_debtors',       # Other debtors / guarantors
    'residence_since',     # Present residence since
    'property',           # Property
    'age',                # Age in years
    'other_installment',  # Other installment plans
    'housing',            # Housing
    'existing_credits',   # Number of existing credits at this bank
    'job',                # Job
    'people_liable',      # Number of people being liable to provide maintenance for
    'telephone',          # Telephone
    'foreign_worker',     # Foreign worker
    'target'              # Target variable (1 = Good, 2 = Bad)
]

# Read the data file
df = pd.read_csv('statlog+german+credit+data/german.data', 
                 sep=' ', 
                 names=columns,
                 header=None)

# Define decoding dictionaries for categorical variables
checking_account_map = {
    'A11': '< 0 DM',
    'A12': '0 <= ... < 200 DM',
    'A13': '>= 200 DM / salary assignments',
    'A14': 'no checking account'
}

credit_history_map = {
    'A30': 'no credits taken/all paid back duly',
    'A31': 'all credits at this bank paid back duly',
    'A32': 'existing credits paid back duly till now',
    'A33': 'delay in paying off in the past',
    'A34': 'critical account/other credits existing'
}

purpose_map = {
    'A40': 'car (new)',
    'A41': 'car (used)',
    'A42': 'furniture/equipment',
    'A43': 'radio/television',
    'A44': 'domestic appliances',
    'A45': 'repairs',
    'A46': 'education',
    'A47': 'vacation',
    'A48': 'retraining',
    'A49': 'business',
    'A410': 'others'
}

savings_account_map = {
    'A61': '< 100 DM',
    'A62': '100 <= ... < 500 DM',
    'A63': '500 <= ... < 1000 DM',
    'A64': '>= 1000 DM',
    'A65': 'unknown/no savings account'
}

employment_map = {
    'A71': 'unemployed',
    'A72': '< 1 year',
    'A73': '1 <= ... < 4 years',
    'A74': '4 <= ... < 7 years',
    'A75': '>= 7 years'
}

personal_status_map = {
    'A91': 'male: divorced/separated',
    'A92': 'female: divorced/separated/married',
    'A93': 'male: single',
    'A94': 'male: married/widowed',
    'A95': 'female: single'
}

other_debtors_map = {
    'A101': 'none',
    'A102': 'co-applicant',
    'A103': 'guarantor'
}

property_map = {
    'A121': 'real estate',
    'A122': 'building society savings agreement/life insurance',
    'A123': 'car or other',
    'A124': 'unknown/no property'
}

other_installment_map = {
    'A141': 'bank',
    'A142': 'stores',
    'A143': 'none'
}

housing_map = {
    'A151': 'rent',
    'A152': 'own',
    'A153': 'for free'
}

job_map = {
    'A171': 'unemployed/unskilled - non-resident',
    'A172': 'unskilled - resident',
    'A173': 'skilled employee/official',
    'A174': 'management/self-employed/highly qualified'
}

telephone_map = {
    'A191': 'none',
    'A192': 'yes, registered under customer name'
}

foreign_worker_map = {
    'A201': 'yes',
    'A202': 'no'
}

# Apply decoding to categorical columns
df['checking_account'] = df['checking_account'].map(checking_account_map)
df['credit_history'] = df['credit_history'].map(credit_history_map)
df['purpose'] = df['purpose'].map(purpose_map)
df['savings_account'] = df['savings_account'].map(savings_account_map)
df['employment'] = df['employment'].map(employment_map)
df['personal_status'] = df['personal_status'].map(personal_status_map)
df['other_debtors'] = df['other_debtors'].map(other_debtors_map)
df['property'] = df['property'].map(property_map)
df['other_installment'] = df['other_installment'].map(other_installment_map)
df['housing'] = df['housing'].map(housing_map)
df['job'] = df['job'].map(job_map)
df['telephone'] = df['telephone'].map(telephone_map)
df['foreign_worker'] = df['foreign_worker'].map(foreign_worker_map)

# Map target values to more meaningful labels
df['target'] = df['target'].map({1: 'Good', 2: 'Bad'})

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Print basic information about the dataset
print("\nDataset Information:")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(X.columns)}")
print("\nFeature types:")
print(X.dtypes)

# Create a figure with subplots for visualizations
plt.figure(figsize=(15, 10))

# 1. Target Distribution
plt.subplot(2, 2, 1)
target_counts = y.value_counts()
plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Credit Risk')

# 2. Age Distribution
plt.subplot(2, 2, 2)
sns.histplot(data=df, x='age', bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')

# 3. Gender Distribution (from personal_status)
plt.subplot(2, 2, 3)
gender_counts = df['personal_status'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution')

# 4. Credit Amount Distribution
plt.subplot(2, 2, 4)
sns.boxplot(data=df, x='target', y='credit_amount')
plt.title('Credit Amount by Risk Level')
plt.xlabel('Credit Risk')
plt.ylabel('Credit Amount')

plt.tight_layout()
plt.savefig('exploratory_analysis.png')
plt.close()

# Print detailed statistics
print("\nDetailed Statistics:")
print("\nAge Statistics:")
print(df['age'].describe())

print("\nCredit Amount Statistics:")
print(df['credit_amount'].describe())

print("\nTarget Distribution:")
print(y.value_counts(normalize=True).round(3))

# Analyze categorical variables
categorical_cols = ['checking_account', 'credit_history', 'purpose', 'savings_account', 
                   'employment', 'personal_status', 'other_debtors', 'property',
                   'other_installment', 'housing', 'job', 'telephone', 'foreign_worker']

print("\nCategorical Variables Distribution:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().head())

# Create a new figure for bias analysis
plt.figure(figsize=(20, 15))

# 1. Gender Bias Analysis
plt.subplot(3, 2, 1)
gender_target = pd.crosstab(df['personal_status'], df['target'], normalize='index') * 100
gender_target.plot(kind='bar', stacked=True)
plt.title('Credit Risk Distribution by Gender and Marital Status')
plt.xlabel('Gender and Marital Status')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Credit Risk')

# 2. Age Bias Analysis
plt.subplot(3, 2, 2)
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                         labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
age_target = pd.crosstab(df['age_group'], df['target'], normalize='index') * 100
age_target.plot(kind='bar', stacked=True)
plt.title('Credit Risk Distribution by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Percentage')
plt.legend(title='Credit Risk')

# 3. Foreign Worker Bias
plt.subplot(3, 2, 3)
foreign_target = pd.crosstab(df['foreign_worker'], df['target'], normalize='index') * 100
foreign_target.plot(kind='bar', stacked=True)
plt.title('Credit Risk Distribution by Foreign Worker Status')
plt.xlabel('Foreign Worker Status')
plt.ylabel('Percentage')
plt.legend(title='Credit Risk')

# 4. Employment Status Bias
plt.subplot(3, 2, 4)
employment_target = pd.crosstab(df['employment'], df['target'], normalize='index') * 100
employment_target.plot(kind='bar', stacked=True)
plt.title('Credit Risk Distribution by Employment Status')
plt.xlabel('Employment Status')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Credit Risk')

# 5. Housing Status Bias
plt.subplot(3, 2, 5)
housing_target = pd.crosstab(df['housing'], df['target'], normalize='index') * 100
housing_target.plot(kind='bar', stacked=True)
plt.title('Credit Risk Distribution by Housing Status')
plt.xlabel('Housing Status')
plt.ylabel('Percentage')
plt.legend(title='Credit Risk')

# 6. Job Level Bias
plt.subplot(3, 2, 6)
job_target = pd.crosstab(df['job'], df['target'], normalize='index') * 100
job_target.plot(kind='bar', stacked=True)
plt.title('Credit Risk Distribution by Job Level')
plt.xlabel('Job Level')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Credit Risk')

plt.tight_layout()
plt.savefig('bias_analysis.png')
plt.close()

# Print detailed bias analysis statistics
print("\nBias Analysis Statistics:")

# Gender and Marital Status Analysis
print("\nGender and Marital Status Distribution:")
gender_stats = pd.crosstab(df['personal_status'], df['target'])
gender_stats['Total'] = gender_stats.sum(axis=1)
gender_stats['Good_Rate'] = (gender_stats['Good'] / gender_stats['Total'] * 100).round(2)
print(gender_stats)

# Age Group Analysis
print("\nAge Group Distribution:")
age_stats = pd.crosstab(df['age_group'], df['target'])
age_stats['Total'] = age_stats.sum(axis=1)
age_stats['Good_Rate'] = (age_stats['Good'] / age_stats['Total'] * 100).round(2)
print(age_stats)

# Foreign Worker Analysis
print("\nForeign Worker Distribution:")
foreign_stats = pd.crosstab(df['foreign_worker'], df['target'])
foreign_stats['Total'] = foreign_stats.sum(axis=1)
foreign_stats['Good_Rate'] = (foreign_stats['Good'] / foreign_stats['Total'] * 100).round(2)
print(foreign_stats)

# Employment Status Analysis
print("\nEmployment Status Distribution:")
employment_stats = pd.crosstab(df['employment'], df['target'])
employment_stats['Total'] = employment_stats.sum(axis=1)
employment_stats['Good_Rate'] = (employment_stats['Good'] / employment_stats['Total'] * 100).round(2)
print(employment_stats)

# Calculate average credit amount by demographic factors
print("\nAverage Credit Amount by Demographic Factors:")
print("\nBy Gender and Marital Status:")
print(df.groupby('personal_status')['credit_amount'].mean().round(2))

print("\nBy Age Group:")
print(df.groupby('age_group')['credit_amount'].mean().round(2))

print("\nBy Foreign Worker Status:")
print(df.groupby('foreign_worker')['credit_amount'].mean().round(2))

# Calculate approval rates
print("\nOverall Approval Rates by Category:")
print("\nGender and Marital Status Approval Rates:")
print((gender_stats['Good'] / gender_stats['Total'] * 100).round(2))

print("\nAge Group Approval Rates:")
print((age_stats['Good'] / age_stats['Total'] * 100).round(2))

print("\nForeign Worker Approval Rates:")
print((foreign_stats['Good'] / foreign_stats['Total'] * 100).round(2))

# Prepare data for modeling
print("\nPreparing data for modeling...")

# Separate numerical and categorical columns
numerical_cols = ['duration', 'credit_amount', 'installment_rate', 'residence_since', 
                 'age', 'existing_credits', 'people_liable']
categorical_cols = ['checking_account', 'credit_history', 'purpose', 'savings_account', 
                   'employment', 'personal_status', 'other_debtors', 'property',
                   'other_installment', 'housing', 'job', 'telephone', 'foreign_worker']

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data
X = df.drop(['target', 'age_group'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Good')
    recall = recall_score(y_test, y_pred, pos_label='Good')
    f1 = f1_score(y_test, y_pred, pos_label='Good')
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'pipeline': pipeline
    }
    
    # Print metrics
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad', 'Good'],
                yticklabels=['Bad', 'Good'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

# Compare models
print("\nModel Comparison:")
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results],
    'Precision': [results[model]['precision'] for model in results],
    'Recall': [results[model]['recall'] for model in results],
    'F1 Score': [results[model]['f1'] for model in results]
})
print(comparison_df)

# Feature importance for Random Forest
if 'Random Forest' in results:
    rf_pipeline = results['Random Forest']['pipeline']
    feature_names = (numerical_cols + 
                    rf_pipeline.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .named_steps['onehot']
                    .get_feature_names_out(categorical_cols))
    
    importances = rf_pipeline.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances - Random Forest')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Save the processed data
X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)

# Print correlation matrix for numerical variables
numerical_cols = ['duration', 'credit_amount', 'installment_rate', 'residence_since', 
                 'age', 'existing_credits', 'people_liable']

plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_cols + ['target']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Analyze fairness metrics by gender
print("\nFairness Analysis by Gender:")

# Get predictions from Random Forest model
rf_pipeline = results['Random Forest']['pipeline']
y_pred = rf_pipeline.predict(X)

# Split data by gender
X_male = X[df['personal_status'].isin(['male: divorced/separated', 'male: single', 'male: married/widowed'])]
X_female = X[df['personal_status'].isin(['female: divorced/separated/married', 'female: single'])]
y_male = y[df['personal_status'].isin(['male: divorced/separated', 'male: single', 'male: married/widowed'])]
y_female = y[df['personal_status'].isin(['female: divorced/separated/married', 'female: single'])]
y_pred_male = rf_pipeline.predict(X_male)
y_pred_female = rf_pipeline.predict(X_female)

# Plot confusion matrices by gender
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm_male = confusion_matrix(y_male, y_pred_male)
sns.heatmap(cm_male, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Male')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
cm_female = confusion_matrix(y_female, y_pred_female)
sns.heatmap(cm_female, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Female')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.savefig('confusion_matrix_by_gender.png')
plt.close()

# Calculate fairness metrics
threshold = 0.1  # Threshold for fairness disparity

# Statistical Parity (Demographic Parity)
pred_pos_rate_male = np.mean(y_pred_male == 'Good')
pred_pos_rate_female = np.mean(y_pred_female == 'Good')
stat_parity_diff = abs(pred_pos_rate_male - pred_pos_rate_female)

# Equal Opportunity
tpr_male = recall_score(y_male, y_pred_male, pos_label='Good')
tpr_female = recall_score(y_female, y_pred_female, pos_label='Good')
equal_opp_diff = abs(tpr_male - tpr_female)

# Equalized Odds
tn_male = confusion_matrix(y_male, y_pred_male)[0][0]
fp_male = confusion_matrix(y_male, y_pred_male)[0][1]
fpr_male = fp_male / (fp_male + tn_male)

tn_female = confusion_matrix(y_female, y_pred_female)[0][0]
fp_female = confusion_matrix(y_female, y_pred_female)[0][1]
fpr_female = fp_female / (fp_female + tn_female)
eq_odds_diff = abs(fpr_male - fpr_female) + abs(tpr_male - tpr_female)

# Predictive Parity
ppv_male = precision_score(y_male, y_pred_male, pos_label='Good')
ppv_female = precision_score(y_female, y_pred_female, pos_label='Good')
pred_parity_diff = abs(ppv_male - ppv_female)

print("\nFairness Metrics (threshold = 0.1):")
print(f"Statistical Parity Difference: {stat_parity_diff:.3f} {'UNFAIR' if stat_parity_diff > threshold else 'FAIR'}")
print(f"Equal Opportunity Difference: {equal_opp_diff:.3f} {'UNFAIR' if equal_opp_diff > threshold else 'FAIR'}")
print(f"Equalized Odds Difference: {eq_odds_diff:.3f} {'UNFAIR' if eq_odds_diff > threshold else 'FAIR'}")
print(f"Predictive Parity Difference: {pred_parity_diff:.3f} {'UNFAIR' if pred_parity_diff > threshold else 'FAIR'}")

# Plot confusion matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm_male = confusion_matrix(y_male, y_pred_male)
sns.heatmap(cm_male, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Male')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
cm_female = confusion_matrix(y_female, y_pred_female)
sns.heatmap(cm_female, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Female')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()
