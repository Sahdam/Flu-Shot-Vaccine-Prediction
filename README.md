# Flu Vaccine Uptake Prediction (H1N1 & Seasonal)

## Project Overview

Can we predict whether individuals received the H1N1 and seasonal flu vaccines using demographic, behavioral, and opinion-based survey data?

This project tackles a multilabel classification problem using data from the National 2009 H1N1 Flu Survey, conducted in the United States following the 2009 H1N1 influenza pandemic. Understanding vaccination behavior is critical for designing effective public health interventions, improving vaccine outreach, and strengthening herd immunity.

## Problem Statement

- Each respondent may receive:

- H1N1 vaccine

- Seasonal flu vaccine

- Both

- Neither

## The goal is to predict the probability that a respondent received:

- h1n1_vaccine

- seasonal_vaccine

**This is a multilabel (not multiclass) prediction task.**


## Dataset Description

Each row represents one survey respondent

Data includes:

- Demographics
- Socioeconomic indicators
- Health status
- Risk perception
- Preventive behaviors
- Doctor recommendations

**Target Variables**
Label	Description
- h1n1_vaccine	Received H1N1 vaccine (0 = No, 1 = Yes)
- seasonal_vaccine	Received seasonal flu vaccine (0 = No, 1 = Yes)

**Feature Summary**

The dataset contains 36 columns:
- respondent_id (unique identifier)
- 35 input features

Feature groups include:

**Opinions & Perceptions**
- h1n1_concern
- h1n1_knowledge
- opinion_h1n1_vacc_effective
- opinion_h1n1_risk
- opinion_h1n1_sick_from_vacc
- opinion_seas_vacc_effective
- opinion_seas_risk
- opinion_seas_sick_from_vacc

**Behavioral Responses**
- behavioral_antiviral_meds
- behavioral_wash_hands
- behavioral_face_mask
- behavioral_avoidance
- behavioral_large_gatherings
- behavioral_outside_home
- behavioral_touch_face

**Health & Access**
- doctor_recc_h1n1
- doctor_recc_seasonal
- chronic_med_condition
- child_under_6_months
- health_worker
- health_insurance

**Demographics & Socioeconomic**
- age_group
- sex
- race
- education
- income_poverty
- marital_status
- rent_or_own
- employment_status
- household_adults
- household_children

**Geographic & Employment**
- hhs_geo_region
- census_msa
- employment_industry
- employment_occupation



## Exploratory Data Analysis (EDA)

Key steps performed:
- Target class balance analysis
- Dependency analysis between the two vaccines
- Correlation heatmaps for numeric features
- Proportional stacked bar plots for:
- Behavioral indicators
- Risk perception
- Socioeconomic factors
- Visualization of vaccination trends across age, income, education, and race


## Modeling Approach
**Problem Framing**
- *Multilabel classification: Two independent but correlated binary targets*


**Preprocessing Pipeline**
**Numerical features:**
- Median imputation
- Standard scaling

**Categorical features:**
- Most-frequent imputation
- One-hot encoding

**These were implemented using ColumnTransformer**

## Models Implemented
**Logistic Regression (Baseline & Tuned)**
- Wrapped in MultiOutputClassifier
- Hyperparameter tuning using GridSearchCV
- Regularization (L1, L2)
- Balanced class weights

**Random Forest (Final Model)**
- MultiOutput Random Forest
- Grid-searched hyperparameters:
    - n_estimators
    - max_depth
    - min_samples_leaf
    - min_samples_split

**Supports both:**
- CPU (scikit-learn)

- GPU (cuML, when available)

## Evaluation Metric
- ROC AUC (Area Under the ROC Curve)
- Computed separately for:
    - h1n1_vaccine
    - seasonal_vaccine
- Final score = mean ROC AUC (macro average)

**Visualizations include:**
- ROC curves per label
- Confusion matrices
- Classification reports

## Key Results

**Random Forest outperformed Logistic Regression**

**Behavioral and opinion-based features were highly predictive**

**Doctor recommendations strongly influenced uptake**

**Age, income, and education played major roles**

 
 Project Structure
├── training_set_features.csv
├── training_set_labels.csv
├── test_set_features.csv
├── submission_1.csv
├── notebook.ipynb
└── README.md


**Final predictions are saved as: submission_1.csv**


## References

- National 2009 H1N1 Flu Survey (CDC)

- Scikit-learn documentation

- Public health literature on vaccine hesitancy



## 👤 Author

**Olayinka Yusuf**
**Data Scientist | Machine Learning | Public Health Analytics**
