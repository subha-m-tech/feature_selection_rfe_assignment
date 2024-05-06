import pandas as pd
from sklearn.model_selection import train_test_split 
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt

def split_scalar(indep_X,dep_Y):
    """This method takes independent and dependent varaibles and split the dataset into training
    and test data"""
    
    X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size = 0.25, random_state = 0)
    
    #Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test
    
def r2_prediction(regressor,X_test,y_test):
    """This method gives R2 score values based on the  test data and model"""
    
    y_pred = regressor.predict(X_test)
    from sklearn.metrics import r2_score
    r2=r2_score(y_test,y_pred)
    return r2
 
def Linear(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create Linear models
    and finally calculate r2 score and returns model object with metrics"""
    

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2   
    
def svm_linear(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_linear models
    and finally calculate r2 score and returns model object with metrics"""
    
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'linear')
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2  
    
def svm_NL(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, create svm_NL models
    and finally calculate r2 score and returns model object with metrics"""
    
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2  
     

def Decision(X_train,y_train,X_test, y_test):
    """This method takes training data and input test data, Decision models
    and finally calculate r2 score and returns model object with metrics"""

    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2  
 

def random_forest(X_train,y_train,X_test, y_test):       
    """This method takes training data and input test data, random forest models
    and finally calculate r2 score and returns model object with metrics"""
    
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)
    r2=r2_prediction(regressor,X_test,y_test)
    return  r2 
    
def rfeFeature(indep_X,dep_Y,n):
    """This method helps to find best features based on n value using RFE-algorithm
    with the help of models"""
    
    rfelist=[]
    
    from sklearn.linear_model import LinearRegression
    lin = LinearRegression()
    
    from sklearn.svm import SVR
    SVRl = SVR(kernel = 'linear')
    
    from sklearn.svm import SVR
    #SVRnl = SVR(kernel = 'rbf')
    
    from sklearn.tree import DecisionTreeRegressor
    dec = DecisionTreeRegressor(random_state = 0)
    
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators = 10, random_state = 0)
    
    
    rfemodellist=[lin,SVRl,dec,rf] 
    for i in   rfemodellist:
        print(i)
        log_rfe = RFE(estimator=i, n_features_to_select=n)
        log_fit = log_rfe.fit(indep_X, dep_Y)
        log_rfe_feature=log_fit.transform(indep_X)
        rfelist.append(log_rfe_feature)
    return rfelist,log_rfe.get_feature_names_out()
    
def rfe_regression(acclog,accsvml,accdes,accrf): 
    """This method returns dataframe with accuracy of different algorithms"""
    
    rfedataframe=pd.DataFrame(index=['Linear','SVC','Random','DecisionTree'],columns=['Linear','SVMl',
                                                                                        'Decision','Random'])

    for number,idex in enumerate(rfedataframe.index):
        
        rfedataframe['Linear'][idex]=acclog[number]       
        rfedataframe['SVMl'][idex]=accsvml[number]
        rfedataframe['Decision'][idex]=accdes[number]
        rfedataframe['Random'][idex]=accrf[number]
    return rfedataframe
