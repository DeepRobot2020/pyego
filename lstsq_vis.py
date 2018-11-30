from scipy.optimize import least_squares
import math as m
import numpy as np
from utils import *


# fit y = exp(m * x + c)

def fun(params, x, y_obs):
    errs = []
    for i, _x in enumerate(x):
        err = y_obs[i] - m.exp(params[0] * _x + params[1])
        errs.append(err)
    return np.array(errs).ravel()


def generate_data(m0, c, noise_scale=1e-2, num_samples=100, n_outliers = 10):
    x = np.random.randn(num_samples, 1)
    y = np.exp(m0 * x + c) +  np.random.randn(num_samples, 1) * noise_scale
    xo = np.random.randint(0, num_samples, n_outliers)
    x[xo] = np.random.randn(len(xo), 1)
    return x, y


LS_PARMS = dict(max_nfev=20, 
                verbose=2,
                x_scale='jac',
                jac='3-point',
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
                method='trf')

params = [0, 0]

x, y = generate_data(0.6, 0.2, 1, 100, 0)
res = least_squares(fun, params, loss='linear', args=(x, y), **LS_PARMS)
cov = covarinace_svd(res.jac)
print('cov', cov.diagonal())
print('est', res.x)

import pdb; pdb.set_trace()
