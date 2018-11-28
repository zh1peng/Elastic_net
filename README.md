# Notes
### Elastic_net_from_sklearn  
Original version is using Elastice net from sklearn.
Elastic net function from Sklearn is super slow compared with glmnet

##
### glmnet_funs_v1

[Glmnet python version](https://glmnet-python.readthedocs.io/en/latest/) was put in the sklearn fashion

## 
### glmnet_funs_v1

self defined parallel cross validation was added to boost the speed


### interpret_results

select and plot significant beta against null model built by permutation.


![sample](https://github.com/zh1peng/Elastic_net/blob/master/SRC_pics/20181128124109.png)

## To do list
1. Choose variable that no need to be normalized and winsorized
2. Multiple resposne (possibly a seperate file)
3. classification (possibly a seperate file)
4. convert vector results into a matrix --- for connectivity prediction
4. Manual on how to use
