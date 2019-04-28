# code to parse permutation code

```python
def collect_results(true_npy, null_npy):
    true_result = np.load(true_npy).item()
    null_result = np.load(null_npy).item()
    return true_result, null_result

def select_sig_betas(true_result, null_result):    
    true_betas=true_result['all_coef'].mean(-1).mean(-1)
    null_betas=null_result['all_coef'].reshape(null_result['all_coef'].shape[0],-1) 
    sig_betas=np.ones(np.size(true_betas))*np.nan
    for idx, val in enumerate (true_betas):
        if val>=np.percentile(null_betas[idx,:],97.5) or val<=np.percentile(null_betas[idx,:],2.5):
            sig_betas[idx]=val
    return sig_betas
def corr_y(y_pred,y):
    if np.ndim(y_pred)==1:
        y_pred=y_pred[:,None]
    y_dim=y.shape[-1]
    r_value=np.empty((y.shape[-1]))*np.nan
    p_value=np.empty((y.shape[-1]))*np.nan
    for y_i in np.arange(y_dim):
        r_value[y_i], p_value[y_i]=scipy.stats.pearsonr(y_pred[:,y_i],y[:,y_i])
    return r_value, p_value
 ```
# example 
```python
true_npy=r'Y:\zhipeng EEG preprocessing\ML_python\ML_activiation_166sphere_1211_only_tlfb_repeat_5_cv_results.npy'
null_npy=r'Y:\zhipeng EEG preprocessing\ML_python\ML_activiation_166sphere_1211_only_tlfb_permutation_5_results.npy'
true_result, null_result=collect_results(true_npy, null_npy)
sig_betas=select_sig_betas(true_result, null_result) 
```
