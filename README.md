# Early-Warning-System-to-Predict-Students-at-Risk-of-Not-Graduating-High-School

Disclaimer: The repository does not contain any data or student information due to data privacy reasons. 

Executive Summary:

This analysis considers graduation outcomes for students from 54 schools under the administration of Muskingum Valley Educational Service Center (MVESC) in South-Eastern Ohio from the academic years of 2006-07 to 2015-16. It identifies key factors in predicting a student’s likelihood of graduation and uses these features to build an early warning system to identify students who are likely not to graduate (at-risk students). This warning system was produced by training and evaluating a large array of machine learning models, using the best performing results according to a rigorous validation process. The resulting models were then analyzed to answer questions such as: which features are most predictive of a student’s failure to graduate? Which student demographics are most likely to exhibit these characteristics, yet perhaps receive disproportionately little support? What is the most efficient way to balance students’ need for academic support with a school administration’s limited budget?

Criteria for selecting best performing model:

1. Top k precision weighted by recency based on performance in multiple train-validation set
2. Performance in fairness metrics such as False discovery rate and True positive rate

The selected predictive models was Logistic Regression with hyperparameters {C: 1, solver: liblinear} which significantly outperform simple baselines in terms of predictive accuracy. If the priority is equity, we suggest using Random Forest with hyperparameters {max_depth: 50, max_features: log2, min_samples_split: 5, n_estimators: 10000} as the model will reduce the disparity in not graduation rates between the protected class (e.g., male) and the reference class (e.g., female).

In particular, our analysis found that male students and students identified as economically disadvantaged are significantly more likely to be at-risk for failure to graduate. In terms of feature importance, our results show that student GPA and the presence or absence of economic disadvantagement are the two strongest predictors of being at-risk of not graduating. Finally, the results of this analysis are used to suggest important policy recommendations for the schools in the studied district. The hope is that implementing such recommendations will lead to more efficient and effective academic interventions and, subsequently, improved student outcomes across all sub-demographics.

Data description:

The data of 61,345 students from MVESC studying in 54 different schools under 13 school districts in the south-eastern Ohio was used in the analysis. The data is longitudinal and it starts from the school year 2006-07, up until 2015-16. In terms of attributes, we have students’ demographic details (such as date of birth, gender, address, whether gifted, socioeconomic status, etc.), academic record (such as subject-wise grades, high-school GPA, Ohio Achievement Assessments and Graduation Tests results, etc.), disciplinary record (absent dates and the nature of absence), and intervention details (i.e., who has been offered intervention and the type of that intervention).


Team members: Justin Jia | Premkumar Loganathan | Elan Rosenfeld | Shirish Verma
