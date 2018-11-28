# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:41:39 2018

@author: Zhipeng
"""

import glmnet_python
from glmnet import glmnet
from cvglmnet import cvglmnet
from cvglmnetPredict import cvglmnetPredict
from cvglmnetPlot import cvglmnetPlot
from cvglmnetCoef import cvglmnetCoef
import pandas as pd
import numpy as np
import sys,time,os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate, cross_val_predict
import scipy.stats
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.utils import shuffle
import scipy

class Winsorizer(BaseEstimator, TransformerMixin):
    """Transforms each feature by clipping from below at the pth quantile
        and from above by the (1-p)th quantile.
     Parameters
    ----------
    quantile : float
        The quantile to clip to.
     copy : boolean, optional, default=True
        Set to False to perform inplace computation during transformation.
     Attributes
    ----------
    quantile : float
        The quantile to clip to.
     data_lb_ : pandas Series, shape (n_features,)
        Per-feature lower bound to clip to.
     data_ub_ : pandas Series, shape (n_features,)
        Per-feature upper bound to clip to.
    """
    def __init__(self, quantile=0.05, copy=True):
        self.quantile = quantile
        self.copy = copy
    def _reset(self):
        """Reset internal data-dependent state of the transformer, if
        necessary. __init__ parameters are not touched.
        """
        if hasattr(self, 'data_lb_'):
            del self.data_lb_
            del self.data_ub_
    def fit(self, X, y=None):
        """Compute the pth and (1-p)th quantiles of each feature to be used
        later for clipping.
         Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to determine clip upper and lower bounds.
         y : Ignored
        """
        X = check_array(X, copy=self.copy, warn_on_dtype=True, estimator=self,
                        dtype=FLOAT_DTYPES)
        self._reset()
        self.data_lb_ = np.percentile(X, 100 * self.quantile, axis=0)
        self.data_ub_ = np.percentile(X, 100 * (1 - self.quantile), axis=0)
        return self
    def transform(self, X):
        """Clips the feature DataFrame X.
         Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to transform.
        """
        check_is_fitted(self, ['data_lb_', 'data_ub_'])
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)
        X = np.clip(X, self.data_lb_, self.data_ub_)
        return X


class glmnet_wrapper(BaseEstimator):
    '''
    Grid search for best alpha and lambda value for ElasticNet 
    with glmnet_python. The main fucntion used is cvglmnet. This 
    is kind of wrapper to put this function in a sklearn fashion,
    which gives more compatibility with sklearn.      
    =======================
    INPUT
    alphas: alpha list to use
    nfold: number of fold to find best lambda per alpha. c.f. cvglmnet
    quantile: persentile to clip for winsorizer (100*quantile = percentail)
    ptype: loss function to use (c.f. cvglmnet)
    OUTPUT ARGUMENTS:
    A dict() is returned with the following fields.
    
    '''
    def __init__(self,alphas=np.linspace(0.1,1,10),nfold=10,quantile=0.01,ptype='mse'):
        self.alphas=alphas
        self.nfold=nfold
        self.quantile=quantile
        self.ptype=ptype
        self.estimators=[]
        self.best_cvm=[]
        self.best_alpha=[]
        self.best_lambda=[]
        self.sd=StandardScaler()
        self.win=Winsorizer(quantile=self.quantile)   
    def fit(self, X, y):
        alphas=self.alphas
        X_norm1=self.sd.fit_transform(X)
        self.win.fit(X_norm1)
        X_norm=self.win.transform(X_norm1)
        
        # covert cv to foldid, keep foldid consistent when comparing between alpha
        cv = KFold(n_splits=self.nfold,shuffle=True)
        foldid2use=y.copy()
        foldid=-1
        for train_index, test_index in cv.split(X):
            foldid+=1
            foldid2use[test_index]=foldid
        foldid2use=foldid2use.astype(int)
        cvms=[]
        lambdas=[]
        for alpha2use in self.alphas:        
            clf_obj=cvglmnet(x = X_norm.copy(),y = y.copy(),
                             foldid=foldid2use,alpha=alpha2use,
                             ptype=self.ptype)            
            self.estimators.append(clf_obj)
            cvms.append(clf_obj['cvm'].min()) 
            # clf_obj['lambda_min']==clf_obj['lambdau'][np.argmin(clf_obj['cvm'])]
            lambdas.append(clf_obj['lambda_min'])
        min_cvm_idx=cvms.index(min(cvms))
        self.best_estimator=self.estimators[min_cvm_idx]
        self.best_cvm=min(cvms)
        self.best_alpha=alphas[min_cvm_idx]
        self.best_lambda=lambdas[min_cvm_idx]
        return self         
    def predict(self, X):
        X_norm1=self.sd.transform(X)
        X_norm=self.win.transform(X_norm1)
        pred=cvglmnetPredict(self.best_estimator, X_norm, s='lambda_min')
        return pred.reshape(-1)
    def get_info(self):
        info={}
        info['best_alpha']=self.best_alpha
        info['best_l1']=self.best_lambda[0]
        info['coef']=cvglmnetCoef(self.best_estimator, s='lambda_min').reshape(-1)
        return info
    def diagnostic_plot1(self,saveto=None):
        fig,axn=plt.subplots(len(self.alphas),1,figsize=(8,60), sharey=True)
        for i, ax in enumerate(axn.flat):
            plt.axes(ax)
            cvglmnetPlot(self.estimators[i])
            textstr='alpha='+str(np.round_(self.alphas[i],3))
            props=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                                            verticalalignment='top', bbox=props)
            plt.tight_layout()
        if saveto!=None:
            plt.savefig(saveto,dpi=300)
    def diagnostic_plot2(self,saveto=None):
        plt.hold(True)
        for idx, alpha2use in enumerate(self.alphas):
            obj2plot=self.estimators[idx]
            col2use=np.random.rand(3,)
            plt.plot(scipy.log(obj2plot['lambdau']),obj2plot['cvm'],color=col2use)
        plt.xlabel('log(lambda)')
        plt.ylabel('Mean-Squared Error')
        plt.title('alpha-lambda-mse')
        plt.legend(np.round_(self.alphas,3),loc='best')  
        if saveto!=None:
            plt.savefig(saveto,dpi=300)

def repeated_cross_validate(clf_in,X,y,cv_fold=10,rep=50,shuffle_y=False):
    '''
    X,y must be np.array
    Do cross validate manually, with following features. 
        1. randomly assign trian and test data for each iteration
        2. averaged beta values for all iterations
        3. averaged pred y for all iterations
        4. save all mse in each fold for all reps
        5. save all best alpha and lambda in each fold for all reps
        6. if shuffle_y=True, then for each iteration, y lable is shuffle. The results are for null model
        Note: this only works with custom classifier 
        zhipeng 2018/09/30
        permutation is added
        zhipeng 2018/10/12   
    '''
    results={}
    all_y_pred=[]
    all_alpha=[]
    all_l1=[]
    all_coef=[]
    all_score=[]
    seed0=np.random.randint(1000,size=1)
    for rep_i in range(rep):
        if shuffle_y:
            seed1=seed0+np.random.randint(100,size=1)
            y2use=shuffle(y.copy(),random_state=int(seed1))
        else:
            y2use=y.copy()   
        seed2=seed0+np.random.randint(100,size=1)
        cv = KFold(n_splits = cv_fold, random_state=int(seed2), shuffle=True)
        fold_coef=[]
        fold_alpha=[]
        fold_l1=[]
        fold_y_pred=np.empty((X.shape[0],))*np.nan
        fold_score=[]
        for train_index, test_index in cv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y2use[train_index], y2use[test_index]
            clf_in.fit(X_train,y_train) # clf_in has standarized function
            y_pred = clf_in.predict(X_test)
            # put y_pred to the same order!
            for idx, val in enumerate(test_index):
                fold_y_pred[val]=y_pred[idx]
#            save pred y, coef, and mse for each main fold
            mse_score=mean_squared_error(y_test, y_pred)
            fold_score.append(mse_score)
#           save parameter tuning results for each main fold            
            tuning_info=clf_in.get_info()
            fold_coef.append(tuning_info['coef']) # coef
            fold_alpha.append(tuning_info['best_alpha'])
            fold_l1.append(tuning_info['best_l1'])
        all_y_pred.append(np.array(fold_y_pred).T)
        all_alpha.append(np.array(fold_alpha).T)
        all_l1.append(np.array(fold_l1).T)
        all_coef.append(np.array(fold_coef))
        all_score.append(np.array(fold_score).T)
    results['all_y_pred']=np.array(all_y_pred).T
    results['mean_y_pred']=np.mean(np.array(all_y_pred),axis=0)
    results['all_alpha']=np.array(all_alpha).T
    results['all_l1']=np.array(all_l1).T
    results['all_coef']=np.array(all_coef).T
    results['all_score']=np.array(all_score).T
    results['true_y']=y.copy()
    return results


def once_csv_pred_test(data_csv):
    csv_name, ext = os.path.splitext(data_csv)
    start_time=time.time()
    all_data=pd.read_csv(data_csv)
    y=np.array(all_data['y'])
    fs=np.array(all_data.drop(columns=['y']))
    clf=glmnet_wrapper()
    y_pred = cross_val_predict(clf, scipy.float64(fs), scipy.float64(y),cv=10,n_jobs=-1)
    r_value, p_value=scipy.stats.pearsonr(y_pred,y)
    e = int(time.time() - start_time)
    e_time='{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
    with open('appending_results.txt','a+') as fo:
        fo.write('\r\n filename---{} r---: {} p---: {} time--: {} \r\n'.format(csv_name, r_value, p_value, e_time))
    return r_value, p_value, e_time

def batch_csv_pred_test(data_path):
    os.chdir(data_path)
    filenames = os.listdir(data_path)
    csv2test=[ filename for filename in filenames if filename.endswith('.csv')]
    all_r=[]
    all_p=[]
    all_time=[]
    for csv in csv2test:
        tmp_r,tmp_p,tmp_time=once_csv_pred_test(csv)
        all_r.append(tmp_r)
        all_p.append(tmp_p)
        all_time.append(tmp_time)
    df=pd.DataFrame([])
    df['r value']=np.array(all_r)
    df['p value']=np.array(all_p)
    df['time']=all_time
    df.index=csv2test
    return df

def repeat_EN_csv(data_csv,reps=50,shuffle_mark=False):
#    repeated_cross_validate(clf_in,X,y,cv_fold=10,rep=50)
    csv_name, ext = os.path.splitext(data_csv)
    all_data=pd.read_csv(data_csv)
    y=np.array((all_data['y']))
    fs=np.array(all_data.drop(columns=['y']))
    results=repeated_cross_validate(glmnet_wrapper(),scipy.float64(fs),scipy.float64(y),cv_fold=10,rep=reps,shuffle_y=shuffle_mark)
    return results

if '__main__'==__name__:
    if len(sys.argv)<3:
        print('Not enough arguement.')
        sys.exit()
    elif sys.argv[1]=='-once':
        start_time=time.time()
        results=once_csv_pred_test(sys.argv[2])
        e = int(time.time() - start_time)
        print(results)
        print('\rTime elapsed:{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)+'\r')
    elif sys.argv[1]=='-batch':
        start_time=time.time()
        data_path=os.path.join(os.getcwd(),sys.argv[2])
        results=batch_csv_pred_test(data_path)
        results.to_csv('batch_test_result.csv')
        e = int(time.time() - start_time)
        print(results)
        print('\rTime elapsed:{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)+'\r')
    elif sys.argv[1]=='-repeat':
        start_time=time.time()
        results=repeat_EN_csv(sys.argv[2],reps=int(sys.argv[3]),shuffle_mark=False)
        e = int(time.time() - start_time)
        print('\rTime elapsed:{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)+'\r')
        np.save('repeat_cv_results.npy', results) 
    elif sys.argv[1]=='-permutation':
        start_time=time.time()
        results=repeat_EN_csv(sys.argv[2],reps=int(sys.argv[3]),shuffle_mark=True)
        e = int(time.time() - start_time)
        print('\rTime elapsed:{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)+'\r')
        np.save('permutation_results.npy', results)



