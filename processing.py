# Import libraries
import pandas as pd
import numpy as np
import logging
from scipy import stats
logger = logging.getLogger(__name__)

# Function desc: train_test_split function splits the data to test and train pairs for each cohort and adds all category levels
#                to both train and test data's categorical columns
# Input: Dataframe with all the cohorts student data
# Return: 1. List with dataframes of test and train data, 2. List of corresponding test and train years for each cohort 
def train_test_split(df):
    # Convert object data type columns to Categorical for entire dataset and fix category
    categorical_columns = [x for x in df.dtypes.index if df.dtypes[x]=='object']
    for col in categorical_columns:
        cat_val = [cat for cat in list(df[col].unique()) if cat is not None]
        df[col] = df[col].astype(pd.CategoricalDtype(categories=cat_val))

    # Split into test and train set
    school_years = np.sort(df["10_school_year"].dropna().unique())[5:-2]

    train_test_pairs = []
    train_test_years = []

    for year in school_years:
        y = year
        while y >= school_years[0]:
            test_df = df[df['10_school_year'] == year].reset_index()
            train_df = df[(df['10_school_year'] >= (y - 3)) & (df['10_school_year'] <= (year - 3))].reset_index()

            train_test_pairs.append([train_df, test_df])
            train_test_years.append([y-3, year])

            y -= 1

    return train_test_pairs, train_test_years

# Function desc: explore_data function provides summary statistics and missing data statistics
# Input: Dataframe with test or train data for a single cohort
# Return: None
def explore_data(df):
    # Summary of numerical variables
    logger.info('Summary of Numerical variables')
    logger.info('------------------------------')
    logger.info(df.describe(include=['int64','float']).transpose())

    # Summary of categorical variables
    logger.info('\nSummary of Categorical variables')
    logger.info('----------------------------------')
    logger.info(df.describe(include=['category']).transpose())

    # Summary of missing values as percentage of total data
    total = df.isnull().sum().sort_values(ascending=False)
    percent = round(((df.isnull().sum()/df.isnull().count())*100),2).sort_values(ascending=False)
    columns_series = df.columns.to_series()

    missing_data_summary_df = pd.concat([columns_series,total, percent],axis = 1, keys=['Column_Names','Total_Missing', 'Percent_Missing'])
    missing_data_summary_df.sort_values(['Percent_Missing'], inplace = True, ascending=False)
    missing_data_summary_df.reset_index(drop=True, inplace=True)
    logger.info('\nMissing data summary')
    logger.info('--------------------')
    logger.info(missing_data_summary_df)

# Function desc: process_data function handles missing data and imputes appropriate values, creates new features
# Input: Train or test dataframe for one cohort
# Return: Dataframe with processed and new features
def process_data(df):

    
    subj_colnames = ['math', 'read', 'science'] 
    selected_colnames = [
            'student_lookup', 'not_graduated', '9_grade_district',
            '10_grade_district', 'gender', 'ethnicity', '9_gpa', 'special_ed',
            'is_gifted','disadvantagement', 'disability',
            'intervention_8_category', 'intervention_9_category',
    ]
    selected_colnames += [f'eighth_{subj}_{suffix}' for subj in subj_colnames for suffix in ['pl', 'ss']]
    df_selected = df[selected_colnames].copy()
    # Converting a numeric column from categorical to int and process missing values for standardized test features
    for subj in subj_colnames:
        ss_col = f'eighth_{subj}_ss'
        df_selected[f'{ss_col}_imputed'] = np.where(df_selected[ss_col].isnull(), 1, 0)
        df_selected[ss_col] = pd.to_numeric(df[ss_col], errors='coerce')
        df_selected[ss_col].fillna(df_selected[ss_col].mean(skipna=True), inplace=True)
        # Add percentiles by school district
        percent_col = f'{subj}_grade_final_percentile_8'
        all_districts_pl = []
        for district in list(df_selected['9_grade_district'].cat.categories):
            district_pl = df_selected[['student_lookup', ss_col]].loc[df_selected['9_grade_district'] == district]
            district_pl[percent_col] = stats.rankdata(district_pl[ss_col], 'max') / len(district_pl)
            all_districts_pl.append(district_pl.drop(columns=ss_col))
        # For students with no district, assign percentile compared to all students
        no_district_pl = df_selected[['student_lookup', ss_col, '9_grade_district']]
        no_district_pl[percent_col] = stats.rankdata(no_district_pl[ss_col], 'max') / len(no_district_pl)
        all_districts_pl.append(no_district_pl[no_district_pl['9_grade_district'].isnull()].drop(columns=[ss_col, '9_grade_district']))
        df_selected = df_selected.join(pd.concat(all_districts_pl).set_index('student_lookup'), on='student_lookup')
        
        # Impute missing values + flag
        pl_col = f'eighth_{subj}_pl'
        cat_list = list(df[pl_col].cat.categories)
        df_selected[f'{pl_col}_imputed'] = np.where(df_selected[pl_col].isnull(), 1, 0)
        mode = df_selected[pl_col].mode(dropna=True)
        # Impute with mode for categorical column
        df_selected[pl_col] = np.where(df_selected[pl_col].isnull(), mode, df_selected[pl_col])
        df_selected[pl_col] = df_selected[pl_col].astype(pd.CategoricalDtype(categories=cat_list))

    # Missing Data imputation for 9th grade GPA and creation of new features  based on 9th GPA 
    for colname in ['9_gpa']:
        df_selected[f'{colname}_imputed'] = np.where(df_selected[colname].isnull(), 1, 0)
        df_selected[colname].fillna(df_selected[colname].mean(), inplace=True)
        percent_col = f'{colname}_percentile'
        all_districts_pl = []
        for district in list(df_selected['9_grade_district'].cat.categories):
            district_pl = df_selected[['student_lookup', colname]].loc[df_selected['9_grade_district'] == district]
            district_pl[percent_col] = stats.rankdata(district_pl[colname], 'max') / len(district_pl)
            all_districts_pl.append(district_pl.drop(columns=colname))
        no_district_pl = df_selected[['student_lookup', colname, '9_grade_district']]
        no_district_pl[percent_col] = stats.rankdata(no_district_pl[colname], 'max') / len(no_district_pl)
        all_districts_pl.append(no_district_pl[no_district_pl['9_grade_district'].isnull()].drop(columns=[colname, '9_grade_district']))
        df_selected = df_selected.join(pd.concat(all_districts_pl).set_index('student_lookup'), on='student_lookup')
    # Missing data imputation for categorical columns 
    df_selected["special_ed"] = np.where(df["special_ed"] == "100", 1, 0)
    df_selected["is_gifted"] = np.where(df["is_gifted"] == 'Y', 1, 0)
    df_selected["disadvantagement"] = np.where(df["disadvantagement"].isnull(), 'no_disadvantagement', df["disadvantagement"])
    df_selected["disability"] = np.where(df["disability"].isnull(), 'no_disability', df["disability"])
    df_selected["student_age"] = np.where(df["student_age"].isnull(), 0, df["student_age"] - df["student_age"].mode(dropna=True)[0])
    df_selected["is_mode_age"] = np.where(df["student_age"] == df["student_age"].mode(dropna=True)[0], 1, 0)
    
    # Processing missing values
    # Mode for gender and ethnicity is the same for all school districts, so just using the overall mode.
    df_selected['gender'] = np.where(df['gender'].isnull(), df['gender'].mode(dropna=True), df['gender'])
    df_selected['gender'] = df_selected['gender'].astype(pd.CategoricalDtype(categories=list(df_selected['gender'].unique())))
    df_selected['ethnicity'] = np.where(df['ethnicity'].isnull(), df['ethnicity'].mode(dropna=True), df['ethnicity'])
    df_selected['ethnicity'] = df_selected['ethnicity'].astype(pd.CategoricalDtype(categories=df['ethnicity'].cat.categories))
    # Missing data imputation for ACS data features
    for demographic_feature in ['Percent_families_BPL', 'Percent_HS_grad_25yrs_above', 'Percent_over_5yrs_not_speak_english_well', 'Percent_with_health_insurance']:
        temp = pd.to_numeric(df[demographic_feature], errors='coerce')
        df_selected[demographic_feature] = temp.fillna(value=temp.mean(skipna=True))
    
    # Missing data imputation for 8th and 9th grade subject scores 
    for class_grade in ['eighth_math_gp', 'eighth_reading_gp', 'ninth_math_gp', 'ninth_reading_gp']:
        df_selected[class_grade] = df[class_grade].fillna(value=df[class_grade].mean(skipna=True))
    
    # Getting categories for both test and train data to avoid category level mismatch for disadvantagement feature
    old_cats = list(df_selected['disadvantagement'].unique())
    new_cats = df['disadvantagement'].cat.categories.tolist()
    cats = pd.Index(list(set(old_cats + new_cats)))
    df_selected['disadvantagement'] = df_selected['disadvantagement'].astype(pd.CategoricalDtype(categories=cats))
    
    # Getting categories for both test and train data to avoid category level mismatch for disability feature
    old_cats = list(df_selected['disability'].unique())
    new_cats = df['disability'].cat.categories.tolist()
    cats = pd.Index(list(set(old_cats + new_cats)))
    df_selected['disability'] = df_selected['disability'].astype(pd.CategoricalDtype(categories=cats))

    # Processing categorical variables:
    logger.info("Processing categorical variables")
    logger.info("--------------------------------")
    
    # Select categorical variables
    categorical_columns = [x for x in df_selected.dtypes.index if df_selected.dtypes[x] not in ('int64','int16','float64','datetime64')]
    logger.info(f'Number of categorical variables: {len(categorical_columns)}')
    logger.info(f'Number of columns in data: {len(df_selected.columns)}')
    # Print frequency of categories
    for col in categorical_columns:
        logger.info(f'\nFrequency of Categories for: {col}')
        logger.info(df_selected[col].value_counts())
    
    # Create dummy variables for categorical variables and remove the categorical data
    dummy_df = pd.get_dummies(df_selected[categorical_columns])
    df_selected = pd.concat([df_selected, dummy_df], axis=1)
    df_selected = df_selected.drop(categorical_columns, axis=1)
    logger.info('\nCategorical variables converted to dummy variables')
    logger.info(f'\nNumber of columns in data after processing categorical variables: {len(df_selected.columns)}')

    return df_selected

# Function desc: create_features function removes columns which are not relevant to features. Most feature engineering is 
#                done in process_data function along with data imputation
# Input: Test or train dataframe for a cohort
# Return: 1. Dataframe with features and 2. series of student Ids
def create_features(df):
    student_ids = df['student_lookup']

    # Dropping label, student ID and school year from feature list
    df.drop(list(df.filter(regex = 'student_lookup')), axis = 1, inplace = True)
    df.drop(list(df.filter(regex = 'school_year')), axis = 1, inplace = True)
    df.drop(list(df.filter(regex = 'not_graduated')), axis = 1, inplace = True)

    feature_df = df.copy()
    return feature_df, student_ids

# Function desc: create_label function seperates the labels from test or train dataframe.  
# Input: test or train dataframe for a cohort
# Return: Series of labels for a test or train dataset
def create_label(df):
    # Label processing and creation is part of the ETL SQL command in the first stage of pipeline itself. 
    # No additional processing required here. 
    
    return df['not_graduated']
