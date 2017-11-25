import numpy as np
import matplotlib.pyplot as plt
from numpy.random import laplace
import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import rpy2
import rpy2.robjects as ro


class Utils(object):

    @staticmethod
    def sub_sampling(point, number_of_samples):
        # the prior distribution uses exponential
        # assume the point are scaled in
        if not point or len(point) < 1:
            raise Exception('the subsampled point is empty')
        if all([e > 1 and e < 0 for e in point]):
            raise Exception('the subsampled point is not within one and zero')
        point_dimension = len(point)
        samples = np.zeros((number_of_samples, len(point)))
        for p in range(number_of_samples):
            within_range = False
            while not within_range:
                cur_sample = laplace(point, decay*np.ones((point_dimension,)) , point_dimension)
                if all([e <= 1 and e >= 0 for e in cur_sample]):
                    within_range = True
            samples[p] = cur_sample
        return samples

    @staticmethod
    def select_feature_lr_wrapper(n_para, x, y, model_type, fit_intercept = False):
        n_r, n_f = x.shape

        if model_type == 'nr':
            general_simple = LogisticRegression()
            general_simple.fit(x, y)
            original_model_paras = general_simple.coef_[0]
            index = np.argsort(abs(original_model_paras))[::-1]
            list_of_select_features = index[:n_para]
            new_x = x[:, list_of_select_features]
            lr_refit = LogisticRegression()
            lr_refit.fit(new_x, y)
            return np.concatenate((lr_refit.coef_[0], lr_refit.intercept_)), 1 - lr_refit.score(new_x, y)

        if model_type == 'bs':
            min_err = float('inf')
            min_ls_f = None
            ls_f_arr = list(combinations(range(n_f), n_para))
            for ls_f in ls_f_arr:
                x_sub = x[:,ls_f]
                general_simple = LogisticRegression(fit_intercept = fit_intercept)
                general_simple.fit(x_sub, y)
                err = 1 - general_simple.score(x_sub, y)
                if err < min_err:
                    min_err = err
                    min_ls_f = ls_f

            x_sub = x[:,min_ls_f]
            lr_refit = LogisticRegression()
            lr_refit.fit(x[:,min_ls_f], y)

            if fit_intercept:
                return np.concatenate((lr_refit.intercept_, lr_refit.coef_[0])), 1 - lr_refit.score(x_sub, y), min_ls_f
            return lr_refit.coef_[0], 1 - lr_refit.score(x_sub, y), min_ls_f

        if model_type == 'vs':

            import rpy2.robjects as ro
            r = ro.r
            from rpy2.robjects.numpy2ri import numpy2ri

            rpy2.robjects.numpy2ri.activate()
            r.library("glmnet")
            r_x = ro.r.assign('dummy', x)
            r_y = ro.r.assign('dummy', y)
            r_fit_intercept = ro.BoolVector((fit_intercept,))

            r(
                '''
                var_select<-function(x,y,degree, fit_intercept){
                fit = glmnet(x, y, intercept = fit_intercept)
                df<-fit$df
                index = which(df<=degree)
                index = index[length(index)]
                lambda = fit$lambda[index]
                coefs = coef(fit, s=lambda)
                if(fit_intercept){
                    res = which(abs(coefs)>0)
                }else{
                    res = which(abs(coefs[2:length(coefs)])>0)
                }
                return(res-1)
                }
                '''
            )
            active_ind = np.asarray(r.var_select(r_x, r_y, n_para, r_fit_intercept[0])).tolist()
            active_ind = [int(x) for x in active_ind]
            lr_refit = LogisticRegression(fit_intercept=fit_intercept)
            lr_refit.fit(x[:, active_ind], y)

            if fit_intercept:
                return np.concatenate((lr_refit.intercept_, lr_refit.coef_[0])), \
                       1 - lr_refit.score(x[:, active_ind], y), active_ind
            return lr_refit.coef_[0], 1 - lr_refit.score(x[:, active_ind], y), active_ind


