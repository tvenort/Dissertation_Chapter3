#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 22:29:12 2023

@author: taishavenort
"""
#Code example
#https://bagheri365.github.io/blog/Tour-of-Machine-Learning-Algorithms-for-Multiclass-Classification/
#https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score, train_test_split
from itertools import combinations
import subprocess
from math import floor
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model, metrics
from sklearn.impute import KNNImputer 
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer
#from sklearn.inspection import plot_partial_dependence

#------Data -------------------------------------------------------------------
data = pd.read_csv('ml_maize_data_final.csv')
data.columns.to_list()  # print column names 
my_list = list(data) # print column names
print(my_list)
print(data)
data = data.rename(columns = {'dist_water_bodies':'Distance.to.river',
                              'richness_tree':'Tree.richness',
                               'Slope':'Soil.slope',
                              'hh_dep_ratio':'Household.dependency.ratio',
                               'dist_forest':'Distance.to.forest',
                               'CIF':'Input.intensity',
                               'Male_Female_ratio': 'Gender.ratio',
                               'leveledu': 'Head.education',
                               'hh_labor_days': 'Labor',
                               'landsize': 'Farm.size',
                               'market_network': 'Market.access',
                               'dist_main_roads':'Distance.to.roads',
                               'wage_entry_n': 'Salary.entry',
                               'liv_sale_number':'Livestock.sales',
                               'Al_sat':'Al.saturation',
                                'C_deficit_indicator':'C.deficit',
                                'N_PNB':'N.PNB',
                                'P_PNB':'P.PNB',
                                'K_PNB': 'K.PNB',
                                'improvedseeds_n':'improved.seeds',
                                'mechtools_n':'mechanized.tools', 
                                'irr_n': 'irrigation',
                                'pestuse_n': 'pesticide.use',
                                'Al_sat':'soil.acidity',
                                'C_capacity':'Carbon.capacity',
                                 'Carbon_storage':'Carbon.storage',
                                'Soil_fertility':'Soil.fertility',
                                'Water_storage':'Water_storage', 
                                'Nutrient_supply':'Nutrient.supply',
                                })
data.columns.to_list()  # print column names 
my_list = list(data) # print column names
print(my_list)
#print(data)
#----Outlier analysis----------------------------------------------------------
# Box Plot
#sns.boxplot(data['Maize_yield']).set(title='SAGCOT Maize yield')
# Position of the Outlier
# the range of maize yield in SSA is between 1.2 and 2.2 t/ha for smallholders
#dropping outliers n= 134 vs n = 153
#outlier_array =np.where(data['Maize_yield']>3500)
#data.drop(data[data['Maize_yield'] >=3500].index, inplace = True)
#print(data)
#-------Feature selection/multicolinearity check with VIF ---------------------
#independent  and dependent variables
dependent_variable = ['Maize_yield_class']
independent_variables = ['Distance.to.river',
                         'Distance.to.forest',
                         'Soil.slope',
                          'Tree.richness',
                           'Gender.ratio',
                            'Household.dependency.ratio',
                             'Labor',
                               'Distance.to.roads',
                                'N.PNB',
                                'P.PNB',
                                'K.PNB',
                                'Input.intensity',
                                ]


iv_data = data[independent_variables]
dv_data = data[dependent_variable]
vif_iv_data = pd.DataFrame()
vif_iv_data["feature"] = iv_data.columns
# calculating VIF for each feature
vif_iv_data["VIF"] = [variance_inflation_factor(iv_data.values, i)
                          for i in range(len(iv_data.columns))]
print(vif_iv_data)

#List of variables to drop vif >= 10 
iv_data.drop([],axis=1,inplace=True) # set axis=0 remove rows, axis =1 to remove columns)
print(iv_data)
#save final dataset
#id_data = ['hh_refno']
#id_data = data[id_data]
data = pd.concat([iv_data, dv_data], axis=1)
print(data)
#data2 = np.asarray(data2, dtype = np.float64)
data.to_csv('data.csv', index=False)

#---Distribution check of selected feature ------------------------------------
# Checking data skewness for assets
data_skew = data.skew().sort_values(ascending=False)
data_skew
# #sq transforms
# def sqr_transform(x):
#     return np.square(x)
# #Log transforms
# def log_transform(x):
#   return np.log1p(x)
#-------Model Setup -----------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  KFold 
X = data[independent_variables]  
y = data[dependent_variable]
n_samples, n_features = X.shape
print('Number of samples:', n_samples)
print('Number of features:', n_features)
target_names = y['Maize_yield_class']

#Cross-validation setup for best parameters search
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,stratify= y,random_state= 123)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

#-------Helper function--------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from matplotlib.colors import ListedColormap
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')

def run_classifier(clf, param_grid, title):
    # -----------------------------------------------------
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    # Randomized grid search
    n_iter_search = 15
    gs = RandomizedSearchCV(clf, 
                            param_distributions = param_grid,
                            n_iter = n_iter_search, 
                            cv = cv, 
                           # iid = False,
                            scoring= 'accuracy')
    # -----------------------------------------------------
    # Train model
    gs.fit(X_train, y_train)  
    print("The best parameters are %s" % (gs.best_params_)) 
    #Predict on train set
    y_pred_train = gs.best_estimator_.predict(X_train)
    # Predict on test set
    y_pred_test = gs.best_estimator_.predict(X_test)
    # Predict on full set
    y_pred = gs.best_estimator_.predict(X)
    # Get Probability estimates on train set
    y_prob_train = gs.best_estimator_.predict_proba(X_train)
    # Get Probability estimates on test set
    y_prob_test = gs.best_estimator_.predict_proba(X_test)
    # Get Probability estimates on full set
    y_prob = gs.best_estimator_.predict_proba(X)
    # -----------------------------------------------------
    #train
    print('Accuracy score train: %.2f%%' %(accuracy_score(y_train, y_pred_train)*100))  
    print('Precision score train: %.2f%%' % (precision_score(y_train, y_pred_train, average= 'weighted')*100))
    print('Recall score train: %.2f%%' % (recall_score(y_train, y_pred_train, average= 'weighted')*100))
    print('F1 score train: %.2f%%' % (f1_score(y_train, y_pred_train, average= 'weighted')*100))

    #test
    print('Accuracy score test: %.2f%%' %(accuracy_score(y_test, y_pred_test)*100))  
    print('Precision score test: %.2f%%' % (precision_score(y_test, y_pred_test, average= 'weighted')*100))
    print('Recall score test: %.2f%%' % (recall_score(y_test, y_pred_test, average= 'weighted')*100))
    print('F1 score test: %.2f%%' % (f1_score(y_test, y_pred_test, average= 'weighted')*100))
    #full
    print('Accuracy score full: %.2f%%' %(accuracy_score(y, y_pred)*100))  
    print('Precision score full: %.2f%%' % (precision_score(y, y_pred, average= 'weighted')*100))
    print('Recall score full: %.2f%%' % (recall_score(y, y_pred, average= 'weighted')*100))
    print('F1 score full: %.2f%%' % (f1_score(y, y_pred, average= 'weighted')*100))
  
    # -----------------------------------------------------
    # Plot confusion matrix
    fig, [ax1,ax2,ax3] = plt.subplots(3, 1, figsize=(7,7),constrained_layout = True)
    cm_test = confusion_matrix(y_test, y_pred_test)#, labels= target_names)
    cm_train = confusion_matrix(y_train, y_pred_train)#, labels= target_names)
    cm_full = confusion_matrix(y, y_pred)#, labels= target_names)
    sns.heatmap(cm_train, annot = True, cbar = False, fmt = "d", linewidths = .5, cmap = "Blues", ax = ax1)
    sns.heatmap(cm_test, annot = True, cbar = False, fmt = "d", linewidths = .5, cmap = "Blues", ax = ax2)
    sns.heatmap(cm_full, annot = True, cbar = False, fmt = "d", linewidths = .5, cmap = "Blues", ax = ax3)
    ax1.set_title("Train")
    ax1.set_xlabel("Predicted class")
    ax1.set_ylabel("Actual class")
    ax2.set_title("Test")
    ax2.set_xlabel("Predicted class")
    ax2.set_ylabel("Actual class")
    ax3.set_title("Full")
    ax3.set_xlabel("Predicted class")
    ax3.set_ylabel("Actual class")
    # ax1.set_xticklabels(target_names)
    # ax1.set_yticklabels(target_names)
    # fig.tight_layout()
    #fig.canvas.draw()
    # labels = [item.get_text() for item in ax1.get_xticklabels()]
    # labels = [item.get_text() for item in ax2.get_xticklabels()]
    # labels = [item.get_text() for item in ax3.get_xticklabels()]
    # labels = [item.get_text() for item in ax1.get_yticklabels()]
    # labels = [item.get_text() for item in ax2.get_yticklabels()]
    # labels = [item.get_text() for item in ax3.get_yticklabels()]
    # labels[1] = 'LP'
    # labels[2] = 'MP'
    # labels[3] = 'HP'

    return gs.best_estimator_, y_pred, y_prob


# #-----Logistic Regression-----------------------
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# param_grid = {'penalty': ['l2'],
#               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
# lr_model, lr_y_pred, lr_y_prob = run_classifier(lr, param_grid, 'Logistic Regression')


# #----Decision Tree-------------------------------
# from sklearn.tree import DecisionTreeClassifier
# dtree = DecisionTreeClassifier()
# param_grid = {'criterion': ['gini', 'entropy'],
#               'splitter': ['best', 'random'],
#               'max_depth': np.arange(1, 20, 2),
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4, 10],
#               'max_features': ['auto', 'sqrt', 'log2', None]}

# dtree_model, dtree_y_pred, dt_y_prob = run_classifier(dtree, param_grid, "Decision Tree")

# # #----Random Forest------------------------------
# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier()

# param_grid = {'n_estimators': [100, 200],
#               'max_depth': [10, 20, 100, None],
#               'max_features': ['auto', 'sqrt', None],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4, 10],
#               'bootstrap': [True, False],
#               'criterion': ['gini', 'entropy']}

# rf_model, rf_y_pred, rf_y_prob = run_classifier(rf, param_grid, 'Random Forest')

#----XGBoost------------------------------
from xgboost import XGBClassifier

xgb = XGBClassifier()

param_grid = {'n_estimators': [100, 200, 300],
             'learning_rate': [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
             'max_depth': [2, 3, 4, 5, 6, 8, 10],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
            'reg_lambda': [0, 1, 3, 5, 7, 9, 11],
            'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}

xgb_model, xgb_y_pred, xgb_y_prob = run_classifier(xgb, param_grid, 'XGBoost')

##--Fitting with best model----------------------------------------------------
#Feature Importance with best model
#retraining with all data
xgb = xgb_model  # Train has been included in the run_classifier function
feature_importance = pd.Series(dict(zip([feature for feature in X.columns.to_list()],
                                        xgb.feature_importances_)))
feature_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.title('Feature Importance')
#Getting model outputs * probabilities with trained model
prob = xgb.predict_proba(X)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('os_prob_pre_int.csv', index=False)
#Getting model outputs * probabilities
probf = xgb.predict(X)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('os_prob_outc_pre_int.csv', index=False)
#saving model as pickke file
# save the XGB model to disk
import joblib
filename = 'finalized_xgb_model.sav'
joblib.dump(xgb, open(filename, 'wb'))
  

#Partial dependence plot
#https://thomasjpfan.github.io/scikit-learn-website/modules/partial_dependence.html
#Partial dependence plot
# features_set1 = ['Soil slope']
# features_set1
# p1=plot_partial_dependence(xgb, X, features_set1, target =0) 
# p2=plot_partial_dependence(xgb, X, features_set1, target =1) 
# p3=plot_partial_dependence(xgb, X, features_set1, target =2) 

###---------------GSA --------------------------------------------------------###
#GSA
### GSA ###
def recalc_close_indices(indices, var):  # subtracts first-order indices from close-form second-order indices
    tmp = indices.filter(regex=var)
    return tmp.sub(indices[var], axis=0)
#import sys
#sys.path.insert(0,'/Library/Frameworks/R.framework/Resources/bin/')
#for p in sys.path: print(p)
#import sys
#print(sys.prefix)
# This part saves .csv files that the R scripts can read. R scripts also write some files
# FIRST ORDER: Create GSA sample
X.to_csv('X.csv', index=False)  # You pass empirical distribution of inputs to GSA code
#New X matrix after identifying distributions 
#X = pd.read_csv('X_dist.csv')
gsa_random_state = 1  # random state for GSA sample matrix
n_gsa_runs = 8500  # 2^n

number_of_classes = data[dependent_variable].nunique().values[0]

for output_i in range(number_of_classes):
    subprocess.call('Rscript --vanilla create_sample.R ' + str(n_gsa_runs) + ' 1 ' + str(2),shell=True)
     # calls R script, 1 = first-order effects
     #outcomes
    gsa_sam = pd.read_csv('gsa_sample.csv')
    gsa_runs = xgb.predict(gsa_sam)  # predictions for GSA sample matrix
    gsa_runs = pd.DataFrame(gsa_runs)
    gsa_runs.to_csv('gsa_runs.csv', index=False) 
    #probabilities
    gsa_sam = pd.read_csv('gsa_sample.csv')
    gsa_runs = xgb.predict_proba(gsa_sam)[:, output_i]  # predictions for GSA sample matrix
    gsa_runs = pd.DataFrame(gsa_runs)
    gsa_runs.to_csv('gsa_runs{}.csv'.format(output_i))
    # FIRST ORDER: GSA of the model
    subprocess.call('Rscript --vanilla analyze_model.R 1', shell=True)  # calls R script, 1 = first-order effects
    first_s_indices_temp = pd.read_csv('S_indices.csv', index_col=0)
    first_s_indices = first_s_indices_temp.loc['original', ]
    bias_si = first_s_indices_temp.loc['bias', ]
    min_ci_si = first_s_indices_temp.loc['min..c.i.', ]
    max_ci_si = first_s_indices_temp.loc['max..c.i.', ]
    # SECOND ORDER: Create GSA sample
    gsa_random_state = 1
    subprocess.call('Rscript --vanilla create_sample.R ' + str(n_gsa_runs) + ' 2 ' + str(gsa_random_state),
                    shell=True)  # calls R script, 2 = second-order effects
    #outcomes
    gsa_sam = pd.read_csv('gsa_sample.csv')
    gsa_runs = xgb.predict(gsa_sam)  # predictions for GSA sample matrix
    gsa_runs = pd.DataFrame(gsa_runs)
    gsa_runs.to_csv('gsa_runs.csv', index=False) 
    #probabilities
    gsa_sam = pd.read_csv('gsa_sample.csv')
    gsa_runs= xgb.predict_proba(gsa_sam)[:, output_i]  # predictions for GSA sample matrix
    gsa_runs = pd.DataFrame(gsa_runs)
    gsa_runs.to_csv('gsa_runs{}.csv'.format(output_i))  
    # SECOND ORDER: GSA of the model
    subprocess.call('Rscript --vanilla analyze_model.R 2', shell=True)  # calls R script, 2 = second-order effects
    second_s_indices_temp = pd.read_csv('S_indices.csv', index_col=0)
    second_s_indices = second_s_indices_temp.loc['original', ]
    # Bias and CIs for the numerical estimation of the indices can be obtained. However, I found this information
    # non-important as the number of GSA runs can be increased without restrictions
    # bias_sij = second_s_indices_temp.loc['bias', ]
    # min_ci_sij = second_s_indices_temp.loc['min..c.i.', ]
    # max_ci_sij = second_s_indices_temp.loc['max..c.i.', ]
    
    comb = [c for c in X.columns.to_list()]  # make combination of variable names for identifying the interactions
    for c in combinations(X.columns, 2):
        comb.append(c[0] + ' X ' + c[1])
    second_s_indices.index = range(len(first_s_indices.index) + 1,
                                   len(first_s_indices.index) + len(second_s_indices.index) + 1)
    
    s_indices = pd.concat([first_s_indices, second_s_indices], axis=0)  # concatenate first and second-order indices
    s_indices.index = comb  # rename indices
    s_indices.clip(lower=0, inplace=True)  # remove S < 0. This can be commented if you want to check for possible errors
    
    n_vars = len(X.columns)  # Number of independent variables
    # This recalculates second-order indices (it subtracts Si and Sj from each Sij)
    for var in s_indices.index.to_list()[0:n_vars]:
        bool_list = []
        [bool_list.append(False) for i in range(n_vars)]  # Create a mask to identify interactions that include a variable
        [bool_list.append(boolean) for boolean in s_indices.index.str.contains(var)[n_vars:]]
        tmp = recalc_close_indices(s_indices, var).drop(var, axis=0)
        s_indices.loc[bool_list] = tmp
        s_indices.clip(lower=0, inplace=True)  # Comment this line to see estimation errors when Sij = 0
    s_indices.to_csv('s_indices_class{}.csv'.format(output_i))
    # bias.to_csv('bias{}.csv'.format(output_i))
    # min_ci.to_csv('min_ci{}.csv'.format(output_i))
    # max_ci.to_csv('max_ci{}.csv'.format(output_i))

# plt.figure(2)
#plt.title(r'{}-fold NSE'.format(k))
#plt.boxplot(r2_cv_score)
#plt.ylabel('NSE')
# plt.savefig('CV_NSE.png')


#outputs comparision
gsa_runs = pd.read_csv('gsa_runs.csv')
runs = pd.read_csv('os_prob_outc_pre_int.csv')
plt.hist(gsa_runs, density=True, alpha=0.5, label='gsa_runs')
plt.hist(runs, density=True, alpha=0.5, label='y observed')
plt.legend()
plt.xlabel('')
plt.ylabel('density')
#plt.savefig('theoreticall_GSA_y_comparison.png')

#distribution check
gsa_sample = pd.read_csv('gsa_sample.csv')
gsa_sample.columns.to_list()  # print column names 
my_list = list(gsa_sample) # print column names
print(my_list)
#k_PNB
X['K.PNB'].hist()
gsa_sample['K.PNB'].hist()
#P_PNB
X['P.PNB'].hist()
gsa_sample['P.PNB'].hist()
#N_PNB
X['N.PNB'].hist()
gsa_sample['N.PNB'].hist()
#Labor
X['Labor'].hist()
gsa_sample['Labor'].hist()
# Household dependency ratio
X['Household.dependency.ratio'].hist()
gsa_sample['Household.dependency.ratio'].hist()
#Gender ratio
X['Gender.ratio'].hist()
gsa_sample['Gender.ratio'].hist()
#Distance to roads
X['Distance.to.roads'].hist()
gsa_sample['Distance.to.roads'].hist()
#Distance to river
X['Distance.to.river'].hist()
gsa_sample['Distance.to.river'].hist()
#Distance to forest
X['Distance.to.forest'].hist()
gsa_sample['Distance.to.forest'].hist()
#soil slope
X['Soil.slope'].hist()
gsa_sample['Soil.slope'].hist()
#Input intensity
X['Input.intensity'].hist()
gsa_sample['Input.intensity'].hist()
#Tree richness
X['Tree.richness'].hist()
gsa_sample['Tree.richness'].hist()

