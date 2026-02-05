# H1N1 and Seasonal Flu Vaccine Prediction Project

## Introduction
This project aims to predict whether individuals will receive the H1N1 vaccine and the seasonal flu vaccine based on various demographic, behavioral, and opinion-based features. The task is a multi-label classification problem, where the model outputs predictions for two distinct target variables: `h1n1_vaccine` and `seasonal_vaccine`.

## Dataset
The dataset consists of three CSV files:
- `training_set_features.csv`: Contains features (independent variables) for the training set, including demographic information, health behaviors, and opinions.
- `training_set_labels.csv`: Contains the labels (dependent variables) for the training set, indicating whether an individual received the H1N1 or seasonal flu vaccine.
- `test_set_features.csv`: Contains features for the test set, for which the final predictions are to be generated.

Key steps in dataset preparation included:
- Loading `training_set_features.csv` and `training_set_labels.csv` into pandas DataFrames.
- Setting `respondent_id` as the index for both DataFrames.
- Merging the feature and label DataFrames into a single `df` for analysis and model training.

## Exploratory Data Analysis (EDA)

### Target Class Balance
An initial analysis of the target variables revealed class imbalances:
- **Seasonal Vaccine**: Approximately 53.5% did not receive the vaccine (`0`) vs. 46.5% received the vaccine (`1`).
- **H1N1 Vaccine**: Approximately 78.8% did not receive the vaccine (`0`) vs. 21.2% received the vaccine (`1`).

### Target Dependency
A `pd.crosstab` analysis showed a noticeable dependency between H1N1 and seasonal vaccine uptake, indicating that individuals who receive one vaccine are more likely to receive the other.

### Feature Analysis
Visualizations (stacked bar plots) were generated to explore the relationship between various features and vaccine uptake for both H1N1 and seasonal flu. Features analyzed included:
- `h1n1_concern`, `h1n1_knowledge`
- Behavioral indicators: `behavioral_antiviral_meds`, `behavioral_wash_hands`, `child_under_6_months`
- Opinion-based features: `opinion_h1n1_vacc_effective`, `opinion_h1n1_risk`, `opinion_h1n1_sick_from_vacc`, `opinion_seas_vacc_effective`, `opinion_seas_risk`, `opinion_seas_sick_from_vacc`
- Demographic features: `income_poverty`, `sex`, `age_group`, `race`

### Missing Values
Several columns contained missing values, with `employment_occupation`, `employment_industry`, and `health_insurance` having the most significant number of `NaN`s. Other columns with missing data included `income_poverty`, `doctor_recc_h1n1`, `doctor_recc_seasonal`, `rent_or_own`, `employment_status`, `marital_status`, `education`, `chronic_med_condition`, `child_under_6_months`, `health_worker`, various opinion features, `household_adults`, `household_children`, behavioral features, `h1n1_knowledge`, `h1n1_concern`, and `census_msa`. These were addressed during preprocessing.

## Data Preprocessing

Data preprocessing was structured using `Pipeline` and `ColumnTransformer` from scikit-learn to ensure reproducibility and proper handling of different feature types.

### Feature Separation
Features were categorized into numerical and categorical types:
- **Numerical Columns**: All columns with `float64` or `int64` Dtype from the original `train_df` (excluding `respondent_id`).
- **Categorical Columns**: `age_group`, `education`, `income_poverty`, `employment_status`.

### Preprocessing Pipelines
- **Numerical Processing**: A pipeline was created for numerical features consisting of:
    - `StandardScaler()`: For scaling numerical features.
    - `SimpleImputer()`: For handling missing numerical values (default strategy is mean).
- **Categorical Processing**: A pipeline was created for categorical features consisting of:
    - `SimpleImputer(strategy="most_frequent")`: For handling missing categorical values.
    - `OneHotEncoder(handle_unknown="ignore")`: For converting categorical features into a one-hot encoded numerical format.

These pipelines were combined using a `ColumnTransformer` to apply the appropriate transformations to their respective feature sets.

## Model Development

The problem was addressed as a multi-label classification task using `MultiOutputClassifier`, which wraps an estimator to handle multiple target variables simultaneously.

### 1. Logistic Regression
- **Baseline Model**: An initial `LogisticRegression` model (`max_iter=2000`) was trained and evaluated.
- **Hyperparameter Tuning**: `GridSearchCV` was employed to find the optimal hyperparameters for `LogisticRegression`. The search space included:
    - `C`: `[0.01, 0.1, 1, 10]`
    - `penalty`: `["l1", "l2"]`
    - `solver`: `["liblinear", "saga"]` (compatible with 'l1' and 'l2' penalties)
    - `class_weight`: `[None, "balanced"]` (to address class imbalance)
- **Best Model (Logistic Regression)**: The best performing Logistic Regression model achieved a `roc_auc_score` of approximately 0.763 on the validation set, with parameters `C=0.1`, `class_weight="balanced"`, `penalty="l1"`, and `solver="saga"`.

### 2. Random Forest Classifier
- **GPU Acceleration**: The project leveraged `cuml.ensemble.RandomForestClassifier` for GPU-accelerated training if CUDA was available, falling back to `sklearn.ensemble.RandomForestClassifier` otherwise, significantly speeding up training on compatible hardware.
- **Hyperparameter Tuning**: `GridSearchCV` was used to optimize `RandomForestClassifier` hyperparameters:
    - `n_estimators`: `[100, 200, 400]`
    - `max_depth`: `[None, 10, 20]`
    - `min_samples_leaf`: `[1, 2, 5]`
    - `min_samples_split`: `[2, 5, 10]`
- **Best Model (Random Forest)**: The best performing Random Forest model achieved a `roc_auc_score` of approximately 0.774 on the validation set. The final chosen model incorporated `class_weight="balanced_subsample"`, `min_samples_leaf=5`, `min_samples_split=5`, `max_depth=10`, and `n_estimators=400`.

## Evaluation Metrics

Model performance was assessed using the following metrics:
- **ROC AUC Score**: The primary evaluation metric, suitable for imbalanced datasets and multi-label classification, as it measures the area under the Receiver Operating Characteristic curve.
- **Classification Report**: Provided detailed metrics (precision, recall, f1-score) for each class of both target variables.
- **Confusion Matrix**: Visualized true positives, true negatives, false positives, and false negatives for each target variable.

## Results

- The Random Forest Classifier consistently outperformed the Logistic Regression model in terms of ROC AUC score on the validation set.
- The best Random Forest model achieved a validation ROC AUC score of approximately 0.774, indicating strong discriminatory power.
- Detailed classification reports and confusion matrices provided insights into the models' performance on predicting each vaccine type.

## Prediction and Submission

The final trained Random Forest model was used to generate probability predictions for the `test_set_features.csv`.
- The `predict_proba` method was used to obtain the probability of vaccination for both H1N1 and seasonal flu.
- These probabilities were saved to `submission_1.csv` in the required format, ready for submission to a competition or for further analysis.

## Technologies Used

- Python
- pandas (for data manipulation and analysis)
- numpy (for numerical operations)
- matplotlib (for plotting and visualization)
- seaborn (for enhanced statistical data visualization)
- scikit-learn (for machine learning models, preprocessing, and evaluation metrics)
    - `train_test_split`, `GridSearchCV`, `ColumnTransformer`, `Pipeline`, `MultiOutputClassifier`, `LogisticRegression`, `RandomForestClassifier`, `StandardScaler`, `SimpleImputer`, `OneHotEncoder`, `roc_auc_score`, `roc_curve`, `classification_report`, `confusion_matrix`
- cuml (for GPU-accelerated Random Forest Classifier, if CUDA is available)
- torch (to detect CUDA availability)
