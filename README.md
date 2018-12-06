# Notes
### Elastic_net_from_sklearn.py 
Original version is using Elastice net from sklearn.
Elastic net function from Sklearn is super slow compared with glmnet

##
### glmnet_funs_v1.py

[Glmnet python version](https://glmnet-python.readthedocs.io/en/latest/) was put in the sklearn fashion

---
### glmnet_funs_v2_parallel.py

self defined parallel cross validation was added to boost the speed

---
### glmnet_funs_v3.py [2018.12.6]

feature added: 

1. allow to specify which features you want to do normalization and winsorization

If using np array, specifiy which columns that need to be excluded. 

Index should be in python style i.e., start from 0

```python
clf=glmnet_wrapper(not2preprocess=[11,12])
```

If using CSV file, using 'flag' in the feature name, then that feature *will not* be normalized and winsorized

2. allow the multiple response target prediction

--
### interpret_results.py

select and plot significant beta against null model built by permutation.

![sample](https://github.com/zh1peng/Elastic_net/blob/master/SRC_pics/20181128124109.png)

---
### Test code
```python
# Test with multi-response [Data provided by glmnet]
import scipy
X = scipy.loadtxt('MultiGaussianExampleX.dat', dtype = scipy.float64, delimiter = ',')
y = scipy.loadtxt('MultiGaussianExampleY.dat', dtype = scipy.float64, delimiter = ',')
clf=glmnet_wrapper()
fold_coef_, fold_alpha_, fold_l1_,fold_score_, sorted_y_pred_=parallel_cv(clf,X,y)
clf.fit(X,y)
clf.diagnostic_plot1()
clf.diagnostic_plot2()
corr_y(sorted_y_pred_, y)

#Test with single response data [Diabetes data]
from sklearn import datasets
diabetes=datasets.load_diabetes()
X=diabetes.data
y=diabetes.target
clf=glmnet_wrapper()
y=y[:,None]
fold_coef_, fold_alpha_, fold_l1_,fold_score_, sorted_y_pred_=parallel_cv(clf,X,y)
clf=glmnet_wrapper()
clf.fit(X,y)
clf.diagnostic_plot1()
clf.diagnostic_plot2()

# Test with Real data [Structual data predict age]
from scipy.io import loadmat
X=loadmat('brain_data.mat')['X']
y=loadmat('brain_data.mat')['y']
```

---
## To do list
~~1. Choose variable that no need to be normalized and winsorized~~

~~2. Multiple resposne (possibly a seperate file)~~

3. classification (possibly a seperate file)

~~4. Manual on how to use~~
