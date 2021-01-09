# Import libraries

import pandas as pd
import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from utils import load_data, db_to_df, df_to_db, bias_metrics
from model import build_model, prediction, evaluate_model, baseline
from processing import train_test_split, explore_data, process_data, create_features, create_label
from sklearn.model_selection import ParameterGrid

import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")

# Function desc: Main function calls all the other functions in the pipeline sequantially to extract data, process data, 
#                Build models and obtain results.
# Input: Filename of the SQL scripts which has the ETL scripts to extract data from Postgres database.
# Return: None
def main(filename):
    print('loading data')
    # Establish database connection 
    with open('/data/groups/schools1/mlpolicylab_fall20_schools1/pipeline/db_info.yaml', 'r') as f:
        db_params = yaml.safe_load(f)['db']

    engine = create_engine('postgres://:@{host}:{port}/{dbname}'.format(
      host=db_params['host'],
      port=db_params['port'],
      dbname=db_params['dbname'],
    ))
    # Load data from database to dataframe
    df = load_data(filename, engine)

    # Split dataframe into train and test data. 
    splits, years_reference = train_test_split(df)
    
    for i, (train_df, test_df) in enumerate(splits):
        print(f'processing split {i}')
        
        # Explore data for each of the cohort
        explore_data(train_df)

        # Process train and test data seperately
        updated_df_train = process_data(train_df)
        updated_df_test = process_data(test_df)
        
        # Upload the test and train data to database for future reference and easy retrival 
        updated_df_train.columns = [col.replace('(','').replace(')','').replace(' ','_').replace('/','_') 
                                    for col in updated_df_train.columns]
        updated_df_test.columns = [col.replace('(','').replace(')','').replace(' ','_').replace('/','_') 
                                   for col in updated_df_test.columns]
        
        
        table_name = timestamp + '_' + str(years_reference[i][1]) + '_' + str(years_reference[i][0])
        
        df_to_db(table_name, 'processed_data', updated_df_train, updated_df_test, engine)
        
        # Retreive test and train data from database
        processed_train, processed_test = db_to_df(table_name,'processed_data', engine)

        updated_df_train_f = processed_train.copy()
        updated_df_train_l = processed_train.copy()
        updated_df_test_f = processed_test.copy()
        updated_df_test_l = processed_test.copy()

        # Create features for test and train data
        features_train, train_student_ids = create_features(updated_df_train_f)
        features_test, test_student_ids = create_features(updated_df_test_f)

        # Create labels
        label_train = create_label(updated_df_train_l)
        label_test = create_label(updated_df_test_l)

        # Concatenating features and labels to save in the database
        train_concat = pd.concat([features_train, label_train], axis = 1, sort = False)
        test_concat = pd.concat([features_test, label_test], axis = 1, sort = False)

        # Calculating baseline rate using grade 9 gpa and base rate
        baseline_precision = baseline(test_concat, years_reference[i])
        base_rate = sum(train_concat.not_graduated)/len(train_concat)

        # Saving and reading from database
        df_to_db(table_name, 'model_data', train_concat, test_concat, engine)
        model_train, model_test = db_to_df(table_name, 'model_data', engine)

        features_train = model_train.iloc[:,:-1]
        label_train = model_train.iloc[:,-1]
        features_test = model_test.iloc[:,:-1]
        label_test = model_test.iloc[:,-1]

        # Build model
        algos = ["Logistic", "SVM", "RandomForest", "DecisionTree"]
        gs_params = {"Logistic": ParameterGrid({'solver': ['lbfgs','liblinear', 'saga'],
                                                'C': [0.001, 0.01, 0.1, 1, 2, 5, 10]}),
                     "SVM": ParameterGrid({'C': [0.01, 1, 2, 5, 10], 'kernel': ['rbf', 'sigmoid']}), 
                     "RandomForest": ParameterGrid({'n_estimators': [30, 50, 100, 500, 1000, 10000], 
                                                    'max_depth': [5, 10, 20, 50], 'min_samples_split': [5, 10, 15],
                                                    'max_features': ['auto', 'log2', 'sqrt']}),
                     "DecisionTree": ParameterGrid({'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 20, 50],
                                                    'min_samples_split': [5, 10, 15]})}
        
        print('performing model grid search')
        for model_name in algos:
            params = gs_params[model_name]
            for param in params:
                model = build_model(features_train, label_train, model_name, param)

                # Perform prediction
                pred_proba_train = prediction(features_train, model)
                pred_proba_test = prediction(features_test, model)
                
                # Convert prediction probabilities to dataframes for further processing
                pred_train_df = pd.DataFrame(pred_proba_train, columns = ['probability'])
                pred_test_df = pd.DataFrame(pred_proba_test, columns = ['probability'])
                
                # Retreive hyperparameters for processing
                hyperparameters = ' '.join(["{}: {}".format(key, param[key]) for key in param.keys()])
                
                pred_train_df['model'], pred_train_df['params'] = model_name, hyperparameters
                pred_test_df['model'], pred_test_df['params'] = model_name, hyperparameters
                
                # Get the prediction scores for test and train data
                predictions_train = pd.concat([train_student_ids, pred_train_df], axis = 1, sort = False)
                predictions_test = pd.concat([test_student_ids, pred_test_df], axis = 1, sort = False)
                
                # Calculate the bias metrics 
                TPR_gender, FDR_gender = bias_metrics(predictions_test, processed_test, 'gender')
                TPR_disadvantagement, FDR_disadvantagement = bias_metrics(predictions_test, processed_test, 'disadvantagement')
                
                # Load the prediction results to database for creating visualizations
                df_to_db(table_name, 'predictions', predictions_train, predictions_test, engine)

                # Evaluate model
                metric = evaluate_model(features_test, label_test, model, model_name, baseline_precision, 
                                       hyperparameters, columns=model_train.columns[:-1])

                # saving results
                df_summary = pd.DataFrame({'test_year': years_reference[i][1], 'train_since': years_reference[i][0], 
                                           'algorithm': model_name, 'hyperparameters': hyperparameters,
                                           'baserate': base_rate, 'baseline': [baseline_precision], 'precision': metric,
                                           'TPR_gender': TPR_gender, 'FDR_gender': FDR_gender,
                                           'TPR_disadvantagement': TPR_disadvantagement, 'FDR_disadvantagement': FDR_disadvantagement})
                df_summary.to_sql(name = timestamp, schema = 'performance_metrics', con = engine, 
                                  if_exists = 'append', index = False)
        

if __name__ == "__main__":
    np.random.seed(0)
    filename = 'sql_script.txt'
    timestamp = str(pd.to_datetime('now').strftime("%Y_%b_%d_%-H_%-M"))
    log_filename = f'logs/pipeline_run_{timestamp}.log'
    print(f'logging to {log_filename}.')
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    main(filename)
