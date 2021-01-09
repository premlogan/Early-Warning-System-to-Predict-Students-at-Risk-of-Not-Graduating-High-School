# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn import svm, tree, ensemble, metrics
from sklearn.linear_model import LogisticRegression
logger = logging.getLogger(__name__)

from utils import top_precision, top_idx, top_accuracy

# Function desc: build_model function trains the model on the train data for different algorithms and hyperparameter combinations
# Input: train dataset features dataframe, label for train dataset, name of algorithm, hyperparameters and top 10% as default
# Return: Trained model
def build_model(features_train, label_train, model_name, param, topP=0.1):
    target = np.array(label_train)
    features = np.array(features_train)
    # Calculate the default classifier train accuracy
    logger.info(f'\nUsing {model_name} model')
    logger.info(f'Number of students: {len(target)}, Not_graduated: {target.sum()}')
    logger.info(f'Constant classifier train accuracy: {max(target.sum(), len(target)-target.sum()) / len(target)}')
    # Train the model
    if model_name =='Logistic':
        clf = LogisticRegression(solver = param['solver'], C = param['C']).fit(features, target)
    elif model_name == 'SVM':
        clf = svm.SVC(C = param['C'], kernel = param['kernel'], probability = True).fit(features, target)
    elif model_name == 'DecisionTree':
        clf = tree.DecisionTreeClassifier(criterion = param['criterion'], max_depth = param['max_depth']).fit(features, target)
    elif model_name == 'RandomForest':
        clf = ensemble.RandomForestClassifier(n_estimators = param['n_estimators'], max_depth = param['max_depth'],
                                              min_samples_split = param['min_samples_split'], 
                                              max_features = param['max_features'], n_jobs = 20, 
                                              random_state = 1).fit(features, target)
    else:
        logger.error(f"Model name \'{model_name}\' unclear ")
        raise Exception(f"Model name \'{model_name}\' unclear ")

    logger.info(f'Train Accuracy: {clf.score(features, target)}')
    return clf

# Function desc: prediction function precits the probability of student not graduating 
# Input: List of features from test dataset, the trained model
# Return: Array of prediction probabilities for not graduating 
def prediction(features_test, model):
    features = np.array(features_test)
    pred_proba = model.predict_proba(features_test)[:,1]
    
    return pred_proba

# Function desc: evaluate_model function calculates precision score and plots precision recall plot
# Input: 1. List of features in test data, 2. Label of test data, 3. Trained model, 4. Name of the algorithm, 5. Baseline score,
#        6. Hyperparameters for the model, 7. Top 10% as default, 3. Optional column names
# Return: Top 10% precision score
def evaluate_model(features_test, label_test, model, model_name, baseline, hyperparameters, topP=0.1, columns=None):
    pred_proba = prediction(features_test, model)

    # Calculate traditional accuracy
    cut_off = 0.5
    pred_labels = [1 if i > cut_off else 0 for i in pred_proba]
    accuracy = metrics.accuracy_score(label_test,pred_labels)
    logger.info(f'\nClassic Test Accuracy: {accuracy}')
    logger.info(f'\nHyperparameters = {hyperparameters}')

    #compute precision amongst riskiest students
    topK = int(topP * len(label_test))
    accuracy = top_precision(label_test, pred_proba, topP=topP)
    logger.info(f'{topP} precision: {accuracy}\tBaseline: {baseline}\tBest possible precision: {min(np.sum(label_test), topK) / topK}')

    #plot precision recall curve
    percents = [0.01 * i for i in range(101)]
    ps = [top_precision(label_test, pred_proba, topP = p) for p in percents]
    rs = [top_accuracy(label_test, pred_proba, topP = p) for p in percents]
    Ks = [int(len(label_test) * p) for p in percents]
    plt.plot(percents, ps, label="Precision")
    plt.plot(percents, rs, label="Recall")
    plt.xlabel("Top Percentile")
    plt.ylabel("Precision/Recall")
    plt.title("Precision vs. Top K for {model_name} Model")
    plt.legend()
    #plt.savefig(f"../viz/precision-recall-topk-{model_name}-{hyperparameters}.png")
    plt.clf()

    if columns is not None:
        cross_tab(columns, features_test, pred_proba, topP=topP)

    return accuracy

# Function desc: cross_tab function calculates the cross tab scores for features with highest importance scores
# Input: 1. List of columns, 2. List of features for test data, 3. Prediction probabilities, 4. Default 10%, 5. Number of crosstabs 
# Return: None
def cross_tab(columns, features_test, pred_proba, topP=0.1, num_tabs=10):
    features_test = np.array(features_test)
    riskiest_idx = top_idx(pred_proba, topP=topP)
    riskiest_features = features_test[riskiest_idx]
    bottom_features = features_test[[i for i in range(len(pred_proba)) if i not in riskiest_idx]]

    riskiest_features = np.mean(riskiest_features, axis=0)
    bottom_features = np.mean(bottom_features, axis=0)

    diff = np.abs(riskiest_features - bottom_features)
    #diff = np.abs(1 - (riskiest_features / bottom_features))
    #diff = np.abs((riskiest_features - bottom_features) / np.std(features_test, axis=0))
    diff_idx = np.flip(np.argsort(diff))

    logger.info("\nCross Tab Results")
    for i in range(num_tabs):
        feature_num = diff_idx[i]
        logger.info(f'Feature = {columns[feature_num]}:\tTopK mean = {riskiest_features[feature_num]}\tBottom mean = {bottom_features[feature_num]}')

# Function desc: baseline function creates precision recall curve for baseline rates calculated using 9th grade GPA
# Input: Test dataframe and year list (cohort) for which the baseline precision recall is calculated
# Return: Precision score
def baseline(df, years):
    # takes the validation dataset and plots a baseline plot using grade 9 gpa

    baseline_df = df[['9_gpa', 'not_graduated']].sort_values('9_gpa').reset_index().drop('index', axis = 1)
    total_drop_outs = sum(baseline_df.not_graduated)
    rnge = [0.01 * i for i in range(101)]

    precision = []
    recall = []
    # Calculate precision and recall for baseline data
    for k in rnge:
        first_k = int(round(k*len(baseline_df), 0))
        precision.append(baseline_df.iloc[:first_k,1].sum(axis = 0)/first_k)
        recall.append(baseline_df.iloc[:first_k,1].sum(axis = 0)/total_drop_outs)
    
    # Plot the precision recall curve for baseline data
    plt.plot(rnge, precision, label = 'Precision')
    plt.plot(rnge, recall, label = 'Recall')
    plt.title('Baseline rates using only 9th grade gpa')
    plt.xticks(np.arange(0.0, 1.01, 0.2))
    plt.yticks(np.arange(0.0, 1.01, 0.2))
    plt.xlabel('Top Percentile')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.savefig(f'../viz/baseline-9gpa-precision-recall-{str(years[1])}-{str(years[0])}.png')
    plt.clf()

    return precision[11]
