# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function desc: load_data function reads SQL query to extract data from Postgres db and loads into dataframe
# Input: .txt file SQL ETL query  and engine with database connection information
# Returns: Dataframe with data extracted from database
def load_data(file, engine):
    query = open(file, "r").read()
    dataset = pd.read_sql(query, engine)
    return dataset

# Function desc: df_to_db function loads the training and test dataset to database for easier retrival.
# Input: Name of the database table to store test and train data, name of schema where table should be created, 
#        train and test dataframe and database engine with database connection information
# Returns: None
def df_to_db(table_name, schema, train_df, test_df, engine):
    train_df['split'] = 'train'
    test_df['split'] = 'test'

    train_df.to_sql(name = table_name, schema = schema, con = engine, if_exists = 'append', index = False)
    test_df.to_sql(name = table_name, schema = schema, con = engine, if_exists = 'append', index = False)

# Function desc: db_to_df function extracts the test and train data from database for a cohort and creates dataframe for processing
# Input: Name of the table with train and test data, name of the scheme, engine with database connection information
# Returns: Dataframe with train data and dataframe with test data for a given cohort
def db_to_df(table_name, schema, engine):
    train_query = '''SELECT * FROM {} WHERE split = 'train' '''.format(schema + '."' + table_name + '"')
    test_query = '''SELECT * FROM {} WHERE split = 'test' '''.format(schema + '."' + table_name + '"')
    
    train_df = pd.read_sql(train_query, engine).drop('split', axis = 1)
    test_df = pd.read_sql(test_query, engine).drop('split', axis = 1)
    
    return train_df, test_df


# Function desc: top_idx function identifies the students with highest risk score based on the predicted probability of not grduating 
# Input: Predicted probability scores for each student and 0.1 as default for getting top 10% of the students at risk of not graduating
# Returns: Index of top 10% of students at risk of not graduating
def top_idx(pred_proba, topP=0.1):
    topK = int(topP * len(pred_proba))
    riskiest_idx = np.argsort(pred_proba)
    # smallest to largest, want largest to smallest
    riskiest_idx = np.flip(riskiest_idx)[:topK]
    return riskiest_idx

# Function desc: top_precision determines the proportion of correctly identified at risk students at some top precision level
# Input: labels of ground truth graduation status, prediction probabilities for each student, percentile for cutoff
# Returns: Precision score
def top_precision(label_test, pred_proba, topP = 0.1):
    topK = int(topP * len(pred_proba))
    riskiest_idx = top_idx(pred_proba, topP=topP)
    riskiest_label = label_test[riskiest_idx]
    return np.sum(riskiest_label) / topK

# Function desc: top_accuracy determines the ratio between the number of students in some top percentile that will not graduate
#                and the number of all positive labels in the test set
# Input: labels of ground truth graduation status, prediction probabilities for each student, percentile for cutoff
# Returns: Accuracy scores 
def top_accuracy(label_test, pred_proba, topP = 0.1):
    riskiest_idx = top_idx(pred_proba, topP=topP)
    riskiest_label = label_test[riskiest_idx]
    return np.sum(riskiest_label) / np.sum(label_test)

# Function desc:bias_metrics function calculates the recall disparity and false discovery rate for gender and disadvantagement
# Input: 1. ,2. ,3. list of features with gender and disadvantagement as values
# Returns: 
def bias_metrics(df_pred, df_processed, feature):

    df = df_pred.merge(df_processed, on = 'student_lookup')
    df = df.sort_values(by = 'probability', ascending = False)
    topP = 0.1
    topK = int(topP * len(df))
    df = df.head(topK)

    if feature == 'gender':
        protect_group_true = df_processed[df_processed.gender_M == 1].not_graduated.sum()
        protect_group_pred = df[df.gender_M == 1].not_graduated.sum()
        reference_group_true = df_processed[df_processed.gender_F == 1].not_graduated.sum()
        reference_group_pred = df[df.gender_F == 1].not_graduated.sum()

        protect_pred = len(df[df.gender_M == 1])
        protect_true = df[df.gender_M == 1].not_graduated.sum()
        protect_wrong = protect_pred - protect_true
        reference_pred = len(df[df.gender_F == 1])
        reference_true = df[df.gender_F == 1].not_graduated.sum()
        reference_wrong = reference_pred - reference_true

    elif feature == 'disadvantagement':
        protect_group_true = df_processed[df_processed.disadvantagement_economic == 1].not_graduated.sum()
        protect_group_pred = df[df.disadvantagement_economic == 1].not_graduated.sum()
        reference_group_true = df_processed[df_processed.disadvantagement_no_disadvantagement == 1].not_graduated.sum()
        reference_group_pred = df[df.disadvantagement_no_disadvantagement == 1].not_graduated.sum()

        protect_pred = len(df[df.disadvantagement_economic == 1])
        protect_true = df[df.disadvantagement_economic == 1].not_graduated.sum()
        protect_wrong = abs(protect_pred - protect_true)
        reference_pred = len(df[df.disadvantagement_no_disadvantagement == 1])
        reference_true = df[df.disadvantagement_no_disadvantagement == 1].not_graduated.sum()
        reference_wrong = abs(reference_pred - reference_true)

    recall_disparity = (protect_group_pred/protect_group_true)/(reference_group_pred/reference_group_true)
    fdr = (protect_wrong/protect_pred)/(reference_wrong/reference_pred)

    return recall_disparity, fdr

