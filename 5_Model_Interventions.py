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
# this data should be the sobol sample 
data = pd.read_csv('gsa_sample.csv')
data.columns.to_list()  # print column names 
my_list = list(data) # print column names
print(my_list)
print(data)
column_names=['maize_yield_class']
run = pd.read_csv('gsa_runs.csv',names=column_names)
run
data =pd.concat([data,run],axis=1)
data
data = data.rename(columns = {'dist_water_bodies':'Distance.to.river',
                              'richness_tree':'Tree.richness',
                               'Slope.x':'Soil.slope',
                              'hh_dep_ratio':'Household.dependency.ratio',
                               'dist_forest':'Distance.to.forest',
                               'CIF':'Input.intensity',
                               'CPF':'Soil.management.intensity',
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
                                'C_capacity':'Carbon.capacity',})
data.columns.to_list()  # print column names 
my_list = list(data) # print column names
print(my_list)
dependent_variable = ['maize_yield_class']
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

X = data[independent_variables]  
y = data[dependent_variable]

#---K_PNB interventions-------------------------------------------------------- 
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
X_new['K.PNB'].hist()
mask = X_new['K.PNB'] > -7 # non-behavioral
sum(mask)
#to_replace_with =np.random.standard_cauchy(100000000)
#to_replace_with = to_replace_with[(to_replace_with > -20) & (to_replace_with< -7)]
to_replace_with =np.random.uniform(-50,-6,sum(mask))
len(to_replace_with)
to_replace_with
to_replace_with = to_replace_with[:sum(mask)]
print(to_replace_with)
len(to_replace_with)
X_new.loc[X_new['K.PNB'] > -7, 'K.PNB'] = to_replace_with
X_new['K.PNB'].hist()  # After that, check if the distribution is now correct
k_pnb = pd.DataFrame(X_new['K.PNB'])
k_pnb = pd.DataFrame(k_pnb, columns = ['K.PNB'])
k_pnb.columns
print(k_pnb)
k_pnb.to_csv('int_dist/k_pnb.csv', index=False)
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_k_pnb.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_k_pnb.csv', index=False)

#---P_PNB interventions-------------------------------------------------------- 
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
X_new['P.PNB'].hist()
mask = X_new['P.PNB'] > -6 #  non-behavioral
sum(mask)
#to_replace_with =np.random.standard_cauchy(10000)
#to_replace_with = to_replace_with[(to_replace_with > -42) & (to_replace_with< -5)]
to_replace_with =np.random.uniform(-20,-10,sum(mask))
len(to_replace_with)
to_replace_with
to_replace_with = to_replace_with[:sum(mask)]
print(to_replace_with)
len(to_replace_with)
X_new.loc[X_new['P.PNB'] > -6, 'P.PNB'] = to_replace_with
X_new['P.PNB'].hist()  # After that, check if the distribution is now correct
p_pnb = pd.DataFrame(X_new['P.PNB'])
p_pnb = pd.DataFrame(p_pnb, columns = ['P.PNB'])
p_pnb.columns
print(p_pnb)
p_pnb.to_csv('int_dist/p_pnb.csv', index=False)
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_p_pnb.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_p_pnb.csv', index=False)

#---N_PNB interventions-------------------------------------------------------- 
#Change distribution
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
X_new['N.PNB'].hist()
mask = X_new['N.PNB'] > -23 #  non-behavioral
sum(mask)
#to_replace_with =np.random.standard_cauchy(100000)
#to_replace_with = to_replace_with[(to_replace_with > -24) & (to_replace_with< -20)]
to_replace_with =np.random.uniform(-50,-23,sum(mask))
len(to_replace_with)
to_replace_with
to_replace_with = to_replace_with[:sum(mask)]
print(to_replace_with)
len(to_replace_with)
X_new.loc[X_new['N.PNB'] > -23, 'N.PNB'] = to_replace_with
X_new['N.PNB'].hist()  # After that, check if the distribution is now correct
n_pnb = pd.DataFrame(X_new['N.PNB'])
n_pnb = pd.DataFrame(n_pnb, columns = ['N.PNB'])
n_pnb.columns
print(n_pnb)
n_pnb.to_csv('int_dist/n_pnb.csv', index=False)
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_n_pnb.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_n_pnb.csv', index=False)

#---Household dependency ratio intervention-----------------------------------------
np.random.seed(45)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
X_new['Household.dependency.ratio'].hist()
mask = X_new['Household.dependency.ratio'] > 0.49 # Condition for non-behavioral
mask
# loc, scale = 0., 1.
# to_replace_with = np.random.laplace(loc, scale, 1000)  # New distribution
# to_replace_with= to_replace_with[(to_replace_with>0.16) & (to_replace_with< 0.49)]
# to_replace_with 
to_replace_with =np.random.random(10000000)
to_replace_with
to_replace_with= to_replace_with[(to_replace_with>0.16) & (to_replace_with< 0.49)]
to_replace_with = to_replace_with[:sum(mask)]
#to_replace_with = np.random.randint(-20,-7, sum(mask))  # New distribution
print(to_replace_with)
len(to_replace_with)
X_new.loc[X_new['Household.dependency.ratio'] > 0.49,'Household.dependency.ratio'] = to_replace_with
#X_new.loc[X_new['Household.dependency.ratio'] > -24, 'Household.dependency.ratio'] = to_replace_with
#X_new['Household.dependency.ratio'][mask] = to_replace_with  # Set old values to new ones
X_new['Household.dependency.ratio'].hist()  # After that, check if the distribution is now correct
hh_dr = pd.DataFrame(X_new['Household.dependency.ratio'])
hh_dr = pd.DataFrame(hh_dr, columns = ['Household.dependency.ratio'])
hh_dr.columns
print(hh_dr)
hh_dr.to_csv('int_dist/hh_dr.csv', index=False)
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_hh_dep.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_hh_dep.csv', index=False)

#---Labor intervention---------------------------------------------
import random
#intervention: replace all non behavioral with values between min and -10
#Change distribution
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
X_new['Labor'].hist()
print(X_new)
mask = X_new['Labor'] < 28 # non-behavioral
mask
to_replace_with =np.random.uniform(28,95, sum(mask)) 
to_replace_with
X_new['Labor'][mask] = to_replace_with  # Set old values to new ones
X_new['Labor'].hist()  # After that, check if the distribution is now correct
hh_labor = pd.DataFrame(X_new['Labor'])
hh_labor = pd.DataFrame(hh_labor, columns = ['Labor'])
hh_labor.columns
print(hh_labor)
hh_labor.to_csv('int_dist/hh_labor.csv', index=False)
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_labor.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_labor.csv', index=False)

#---Distance to river intervention-----------------------------------------
#non-behavioral : >-10
import random
#intervention: replace all non behavioral with values between min and -10
#Change distribution
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
X_new['Distance.to.river'].hist()
print(X_new)
mask = X_new['Distance.to.river'] < 11.7  # Condition for non-behavioral
mask
to_replace_with =np.random.uniform(11.7,22, sum(mask)) 
to_replace_with
len(to_replace_with)
X_new['Distance.to.river'][mask] = to_replace_with  # Set old values to new ones
X_new['Distance.to.river'].hist()  # After that, check if the distribution is now correct
dr = pd.DataFrame(X_new['Distance.to.river'])
dr = pd.DataFrame(dr, columns = ['Distance.to.river'])
dr.columns
print(dr)
dr.to_csv('int_dist/int_dist_river.csv', index=False)
#UA
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_dist_riv.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_dist_riv.csv', index=False)

#---Distance to roads-----------------------------------------
#non-behavioral : >-10
import random
#intervention: replace all non behavioral with values between min and -10
#Change distribution
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
X_new['Distance.to.roads'].hist()
print(X_new)
mask = X_new['Distance.to.roads'] >3.7  # Condition for non-behavioral
mask
to_replace_with =np.random.uniform(0,3.7, sum(mask)) 
to_replace_with
len(to_replace_with)
X_new['Distance.to.roads'][mask] = to_replace_with  # Set old values to new ones
X_new['Distance.to.roads'].hist()  # After that, check if the distribution is now correct
drd = pd.DataFrame(X_new['Distance.to.roads'])
drd = pd.DataFrame(drd, columns = ['Distance.to.roads'])
drd.columns
print(drd)
drd.to_csv('int_dist/int_dist_roads.csv', index=False)
#UA
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_dist_roads.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_dist_roads.csv', index=False)

#---Slope intervention -----------------------------------------
# #non-behavioral : >-10
import random
#intervention: replace all non behavioral with values between min and -10
#Change distribution
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
X_new['Soil.slope'].hist()
print(X_new)
mask = X_new['Soil.slope'] < 2.5  # Condition for non-behavioral
mask
to_replace_with =np.random.uniform(2.5,13, sum(mask)) 
X_new['Soil.slope'][mask] = to_replace_with  # Set old values to new ones
X_new['Soil.slope'].hist()  # After that, check if the distribution is now correct
slope = pd.DataFrame(X_new['Soil.slope'])
slope = pd.DataFrame(slope, columns = ['Soil.slope'])
slope.columns
print(slope)
slope.to_csv('int_dist/soil_slope.csv', index=False)
#UA
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_slope.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_slope.csv', index=False)

#--------Input intensification intervention-----------------------------------
# #non-behavioral : < 2.4
import random
#intervention: replace all non behavioral with values between min and -10
#Change distribution
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
X_new['Input.intensity'].hist()
print(X_new)
mask = X_new['Input.intensity'] < 1.2  # Condition for non-behavioral
mask
to_replace_with =np.random.randint(1.2,14, sum(mask)) 
X_new['Input.intensity'][mask] = to_replace_with  # Set old values to new ones
X_new['Input.intensity'].hist()  # After that, check if the distribution is now correct
Input_intensity = pd.DataFrame(X_new['Input.intensity'])
Input_intensity = pd.DataFrame(Input_intensity, columns = ['Input.intensity'])
Input_intensity.columns
print(Input_intensity)
Input_intensity.to_csv('int_dist/Input_intensity.csv', index=False)
#UA
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_input_int.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_input_int.csv', index=False)

#Fertilizer interventions ----------------------------------------------------
#non-behavioral : >-10
#intervention: replace all non behavioral with values between min and -10
#Change distribution
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
print(X_new)
#N_PNB
to_replace_with = pd.read_csv('int_dist/n_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['N.PNB'] = to_replace_with
X_new['N.PNB'].hist()
#P_PNB
to_replace_with = pd.read_csv('int_dist/p_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['P.PNB'] = to_replace_with
X_new['P.PNB'].hist()
#K_PNB
to_replace_with = pd.read_csv('int_dist/k_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['K.PNB'] = to_replace_with
X_new['K.PNB'].hist()
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_fert.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_fert.csv', index=False)


#-----infrastructure interventions---------------------------------------------
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
print(X_new)
# #Dist to river
to_replace_with = pd.read_csv('int_dist/int_dist_river.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Distance.to.river'] = to_replace_with
X_new['Distance.to.river'].hist()
# #Dist to roads
to_replace_with = pd.read_csv('int_dist/int_dist_roads.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Distance.to.roads'] = to_replace_with
X_new['Distance.to.roads'].hist()
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_linf.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_linf.csv', index=False)

#-----household interventions-----------
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
print(X_new)
#Input intensity
to_replace_with = pd.read_csv('int_dist/Input_intensity.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Input.intensity'] = to_replace_with
X_new['Input.intensity'].hist()
#Labor
to_replace_with = pd.read_csv('int_dist/hh_labor.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Labor'] = to_replace_with
X_new['Labor'].hist()
#dependency ratio
to_replace_with = pd.read_csv('int_dist/hh_dr.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Household.dependency.ratio'] = to_replace_with
X_new['Household.dependency.ratio'].hist()
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_hhc.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_hhc.csv', index=False)

#-----K and soil slope interventions ----------------------------------
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
print(X_new)
#Soil slope
to_replace_with = pd.read_csv('int_dist/soil_slope.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Soil.slope'] = to_replace_with
X_new['Soil.slope'].hist()
# #N_PNB
# to_replace_with = pd.read_csv('int_dist/n_pnb.csv')
# to_replace_with = to_replace_with.to_numpy()
# to_replace_with
# X_new['N.PNB'] = to_replace_with
# X_new['N.PNB'].hist()
# #P_PNB
# to_replace_with = pd.read_csv('int_dist/p_pnb.csv')
# to_replace_with = to_replace_with.to_numpy()
# to_replace_with
# X_new['P.PNB'] = to_replace_with
# X_new['P.PNB'].hist()
#K_PNB
to_replace_with = pd.read_csv('int_dist/k_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['K.PNB'] = to_replace_with
X_new['K.PNB'].hist()
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_k_ss.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_k_ss.csv', index=False)
   
#-----Soil health interventions ----------------------------------
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
print(X_new)
#Soil slope
to_replace_with = pd.read_csv('int_dist/soil_slope.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Soil.slope'] = to_replace_with
X_new['Soil.slope'].hist()
#N_PNB
to_replace_with = pd.read_csv('int_dist/n_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['N.PNB'] = to_replace_with
X_new['N.PNB'].hist()
#P_PNB
to_replace_with = pd.read_csv('int_dist/p_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['P.PNB'] = to_replace_with
X_new['P.PNB'].hist()
#K_PNB
to_replace_with = pd.read_csv('int_dist/k_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['K.PNB'] = to_replace_with
X_new['K.PNB'].hist()
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_sh.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_sh.csv', index=False)

#soil health and household interventions ------------------------------------------------------
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
print(X_new)
#Soil slope
to_replace_with = pd.read_csv('int_dist/soil_slope.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Soil.slope'] = to_replace_with
X_new['Soil.slope'].hist()
#N_PNB
to_replace_with = pd.read_csv('int_dist/n_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['N.PNB'] = to_replace_with
X_new['N.PNB'].hist()
#P_PNB
to_replace_with = pd.read_csv('int_dist/p_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['P.PNB'] = to_replace_with
X_new['P.PNB'].hist()
#K_PNB
to_replace_with = pd.read_csv('int_dist/k_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['K.PNB'] = to_replace_with
X_new['K.PNB'].hist()
#Input intensity
to_replace_with = pd.read_csv('int_dist/Input_intensity.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Input.intensity'] = to_replace_with
X_new['Input.intensity'].hist()
#Labor
to_replace_with = pd.read_csv('int_dist/hh_labor.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Labor'] = to_replace_with
X_new['Labor'].hist()
#dependency ratio
to_replace_with = pd.read_csv('int_dist/hh_dr.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Household.dependency.ratio'] = to_replace_with
X_new['Household.dependency.ratio'].hist()
# #Dist to river
# to_replace_with = pd.read_csv('int_dist/int_dist_river.csv')
# to_replace_with = to_replace_with.to_numpy()
# to_replace_with
# X_new['Distance.to.river'] = to_replace_with
# X_new['Distance.to.river'].hist()
# #Dist to roads
# to_replace_with = pd.read_csv('int_dist/int_dist_roads.csv')
# to_replace_with = to_replace_with.to_numpy()
# to_replace_with
# X_new['Distance.to.roads'] = to_replace_with
# X_new['Distance.to.roads'].hist()
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_sh_hh.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_sh_hh.csv', index=False)

#-----soil health, household,and landscape infrastructure combined------------
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
print(X_new)
#Soil slope
to_replace_with = pd.read_csv('int_dist/soil_slope.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Soil.slope'] = to_replace_with
X_new['Soil.slope'].hist()
#N_PNB
to_replace_with = pd.read_csv('int_dist/n_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['N.PNB'] = to_replace_with
X_new['N.PNB'].hist()
#P_PNB
to_replace_with = pd.read_csv('int_dist/p_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['P.PNB'] = to_replace_with
X_new['P.PNB'].hist()
#K_PNB
to_replace_with = pd.read_csv('int_dist/k_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['K.PNB'] = to_replace_with
X_new['K.PNB'].hist()
#Input intensity
to_replace_with = pd.read_csv('int_dist/Input_intensity.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Input.intensity'] = to_replace_with
X_new['Input.intensity'].hist()
#Labor
to_replace_with = pd.read_csv('int_dist/hh_labor.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Labor'] = to_replace_with
X_new['Labor'].hist()
#dependency ratio
to_replace_with = pd.read_csv('int_dist/hh_dr.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Household.dependency.ratio'] = to_replace_with
X_new['Household.dependency.ratio'].hist()
# #Dist to river
to_replace_with = pd.read_csv('int_dist/int_dist_river.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Distance.to.river'] = to_replace_with
X_new['Distance.to.river'].hist()
# #Dist to roads
to_replace_with = pd.read_csv('int_dist/int_dist_roads.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Distance.to.roads'] = to_replace_with
X_new['Distance.to.roads'].hist()
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_sh_hh_ls.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_sh_hh_ls.csv', index=False)
      
#---All interventions combined interventions ----------------------------------
np.random.seed(42)
#X['K.PNB_new'] = np.where(X['K.PNB'] > -10,np.random.randint(-40, -10),X.shape[0])
X_new = X.copy()
print(X_new)
#Soil slope
to_replace_with = pd.read_csv('int_dist/soil_slope.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Soil.slope'] = to_replace_with
X_new['Soil.slope'].hist()
#Dist to river
to_replace_with = pd.read_csv('int_dist/int_dist_river.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Distance.to.river'] = to_replace_with
X_new['Distance.to.river'].hist()
#Dist to roads
to_replace_with = pd.read_csv('int_dist/int_dist_roads.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Distance.to.roads'] = to_replace_with
X_new['Distance.to.roads'].hist()
#Labor
to_replace_with = pd.read_csv('int_dist/hh_labor.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Labor'] = to_replace_with
X_new['Labor'].hist()
#dependency ratio
to_replace_with = pd.read_csv('int_dist/hh_dr.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Household.dependency.ratio'] = to_replace_with
X_new['Household.dependency.ratio'].hist()
#Input intensity
to_replace_with = pd.read_csv('int_dist/Input_intensity.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['Input.intensity'] = to_replace_with
X_new['Input.intensity'].hist()
#N_PNB
to_replace_with = pd.read_csv('int_dist/n_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['N.PNB'] = to_replace_with
X_new['N.PNB'].hist()
#P_PNB
to_replace_with = pd.read_csv('int_dist/p_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['P.PNB'] = to_replace_with
X_new['P.PNB'].hist()
#K_PNB
to_replace_with = pd.read_csv('int_dist/k_pnb.csv')
to_replace_with = to_replace_with.to_numpy()
to_replace_with
X_new['K.PNB'] = to_replace_with
X_new['K.PNB'].hist()
#Fitting new X distribution in model
import joblib
# load the saved model
filename ='finalized_xgb_model.sav'
model = joblib.load(open(filename, 'rb'))
#new_y_distribution = model.predict(X_new)  # X with the new distribution or predict_proba
#Getting model outputs * probabilities with trained model
prob = model.predict_proba(X_new)
prob = pd.DataFrame(prob, columns = ['Class_O','Class_1','Class_2'])
print(prob)
prob.to_csv('post_int_prob/prob_all.csv', index=False)
#Getting model outputs * probabilities of intervention
probf = model.predict(X_new)
probf = pd.DataFrame(probf, columns = ['Class'])
print(probf)
probf.to_csv('post_int_prob/outcome_all.csv', index=False)
   
   
