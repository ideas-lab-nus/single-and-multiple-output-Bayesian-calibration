"""
Multiple-output Bayesian calibration for energy modelling
"""
__author__ = "Yaonan Gu, Chao Song"
__copyright__ = "Copyright 2023, National University of Singapore - NUS"
__credits__ = [""]
__license__ = "To be added"
__version__ = "3"
__maintainer__ = "Yaonan Gu, Chao Song"
__email__ = "yaonangu@u.nus.edu, nussongchao@gmail.com"
__status__ = "Experimental/Beta"

import numpy as np
np.random.seed(206)

import yaml
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import theano
import theano.tensor as tt
import pymc3 as pm
from pymc3.gp.cov import Covariance
import scipy.stats as st
import random

class MultiMarginal(pm.gp.gp.Base):
    R"""
    MultiMarginal Gaussian process.
    The `MultiMarginal` class is an implementation of the sum of a GP
    prior and additive noise.  It has `marginal_likelihood`, `conditional`
    and `predict` methods.  This GP implementation can be used to
    implement regression on data that is normally distributed.  For more
    information on the `prior` and `conditional` methods, see their docstrings.
    Parameters
    ----------
    cov_func: None, 2D array, or instance of Covariance
        The covariance function.  Defaults to zero.
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.
    Examples
    --------
    .. code:: python
        # A one dimensional column vector of inputs.
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            # Specify the covariance function.
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)
            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.Marginal(cov_func=cov_func)
            # Place a GP prior over the function f.
            sigma = pm.HalfCauchy("sigma", beta=3)
            y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)
        ...
        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        Xnew = np.linspace(-1, 2, 50)[:, None]
        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)
    """

    def _build_marginal_likelihood(self, X, y, noise):
        mu = tt.zeros_like(y) # self.mean_func(X)
        Kxx = self.cov_func(X)
        Knx = noise(X)
        cov = Kxx + Knx
        return mu, cov

    def marginal_likelihood(self, name, X, y, colchol, noise, matrix_shape, is_observed=True, **kwargs):
        R"""
        Returns the marginal likelihood distribution, given the input
        locations `X` and the data `y`.
        This is integral over the product of the GP prior and a normal likelihood.
        .. math::
           y \mid X,\theta \sim \int p(y \mid f,\, X,\, \theta) \, p(f \mid X,\, \theta) \, df
        Parameters
        ----------
        name: string
            Name of the random variable
        X: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        y: array-like
            Data that is the sum of the function with the GP prior and Gaussian
            noise.  Must have shape `(n, )`.
        noise: scalar, Variable, or Covariance
            Standard deviation of the Gaussian noise.  Can also be a Covariance for
            non-white noise.
        is_observed: bool
            Whether to set `y` as an `observed` variable in the `model`.
            Default is `True`.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        if not isinstance(noise, Covariance):
            noise = pm.gp.cov.WhiteNoise(noise)
        mu, cov = self._build_marginal_likelihood(X, y, noise)
        self.X = X
        self.y = y
        self.noise = noise

        # Warning: the shape of y is hardcode

        if is_observed:
            return pm.MatrixNormal(name, mu=mu, colchol=colchol, rowcov=cov, observed=y, shape=(matrix_shape[0],matrix_shape[1]), **kwargs)
        else:
            shape = infer_shape(X, kwargs.pop("shape", None))
            return pm.MvNormal(name, mu=mu, cov=cov, shape=shape, **kwargs)

    def _get_given_vals(self, given):
        if given is None:
            given = {}

        if 'gp' in given:
            cov_total = given['gp'].cov_func
            mean_total = given['gp'].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ['X', 'y', 'noise']):
            X, y, noise = given['X'], given['y'], given['noise']
            if not isinstance(noise, Covariance):
                noise = pm.gp.cov.WhiteNoise(noise)
        else:
            X, y, noise = self.X, self.y, self.noise
        return X, y, noise, cov_total, mean_total

    def _build_conditional(self, Xnew, pred_noise, diag, X, y, noise,
                           cov_total, mean_total):
        Kxx = cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        Knx = noise(X)
        rxx = y - mean_total(X)
        L = cholesky(stabilize(Kxx) + Knx)
        A = solve_lower(L, Kxs)
        v = solve_lower(L, rxx)
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), v)
        if diag:
            Kss = self.cov_func(Xnew, diag=True)
            var = Kss - tt.sum(tt.square(A), 0)
            if pred_noise:
                var += noise(Xnew, diag=True)
            return mu, var
        else:
            Kss = self.cov_func(Xnew)
            cov = Kss - tt.dot(tt.transpose(A), A)
            if pred_noise:
                cov += noise(Xnew)
            return mu, cov if pred_noise else stabilize(cov)

    def conditional(self, name, Xnew, pred_noise=False, given=None, **kwargs):
        R"""
        Returns the conditional distribution evaluated over new input
        locations `Xnew`.
        Given a set of function values `f` that the GP prior was over, the
        conditional distribution over a set of new points, `f_*` is:
        .. math::
           f_* \mid f, X, X_* \sim \mathcal{GP}\left(
               K(X_*, X) [K(X, X) + K_{n}(X, X)]^{-1} f \,,
               K(X_*, X_*) - K(X_*, X) [K(X, X) + K_{n}(X, X)]^{-1} K(X, X_*) \right)
        Parameters
        ----------
        name: string
            Name of the random variable
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Can optionally take as key value pairs: `X`, `y`, `noise`,
            and `gp`.  See the section in the documentation on additive GP
            models in PyMC3 for more information.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        givens = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, pred_noise, False, *givens)
        shape = infer_shape(Xnew, kwargs.pop("shape", None))
        return pm.MvNormal(name, mu=mu, cov=cov, shape=shape, **kwargs)

    def predict(self, Xnew, point=None, diag=False, pred_noise=False, given=None):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as numpy arrays, given a `point`, such as the MAP
        estimate or a sample from a `trace`.
        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        point: pymc3.model.Point
            A specific point to condition on.
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Same as `conditional` method.
        """
        if given is None:
            given = {}

        mu, cov = self.predictt(Xnew, diag, pred_noise, given)
        return draw_values([mu, cov], point=point)

    def predictt(self, Xnew, diag=False, pred_noise=False, given=None):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as symbolic variables.
        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Same as `conditional` method.
        """
        givens = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, pred_noise, diag, *givens)
        return mu, cov


# Function for Bayesian calibration
def MultiOutput_Bayesian_Calibration(n_y,DataComp,DataField,DataPred,output_folder):
    # Data preprocessing
    n = np.shape(DataField)[0] # number of measured data
    m = np.shape(DataComp)[0] # number of simulation data

    p = np.shape(DataField)[1] - n_y # number of input x
    q = np.shape(DataComp)[1] - p - n_y # number of calibration parameters t

    xc = DataComp[:,n_y:] # simulation input x + calibration parameters t
    xf = DataField[:,n_y:] # observed input

    yc = DataComp[:,:n_y] # simulation output
    yf = DataField[:,:n_y] # observed output

    x_pred = DataPred[:,n_y:] # designed points for predictions
    y_true = DataPred[:,:n_y] # true values for designed points for predictions
    n_pred = np.shape(x_pred)[0] # number of predictions
    N = n+m+n_pred

    # Put points xc, xf, and x_pred on [0,1] 
    for i in range(p):
        x_min = min(min(xc[:,i]),min(xf[:,i]))
        x_max = max(max(xc[:,i]),max(xf[:,i]))
        xc[:,i] = (xc[:,i]-x_min)/(x_max-x_min)
        xf[:,i] = (xf[:,i]-x_min)/(x_max-x_min)
        x_pred[:,i] = (x_pred[:,i]-x_min)/(x_max-x_min)

    # Put calibration parameters t on domain [0,1]
    for i in range(p,(p+q)):
        t_min = min(xc[:,i])
        t_max = max(xc[:,i])
        xc[:,i] = (xc[:,i]-t_min)/(t_max-t_min)

    # Store mean and std of yc for future use of scaling back 
    yc_mean = np.zeros(n_y)
    yc_sd = np.zeros(n_y)

    # Standardization of output yf and yc
    for i in range(n_y):
        yc_mean[i] = np.mean(yc[:,i])
        yc_sd[i] = np.std(yc[:,i])
        yc[:,i] = (yc[:,i]-yc_mean[i])/yc_sd[i]
        yf[:,i] = (yf[:,i]-yc_mean[i])/yc_sd[i]

    # PyMC3 modelling
    with pm.Model() as model:
        # Set priors
        eta1 = pm.HalfCauchy("eta1", beta=5) # Set eta of gaussian process
        lengthscale = pm.Gamma("lengthscale", alpha=2, beta=1, shape=(p+q)) # Set lengthscale of gaussian process
        tf = pm.Beta("tf", alpha=2, beta=2, shape=q) # Set for calibration parameters
        sigma1 = pm.HalfCauchy('sigma1', beta=5) # Set for noise
        y_pred = pm.Normal('y_pred', 0, 1.5, shape=(n_pred,n_y)) # y prediction

        # Setup prior of right cholesky matrix
        sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=n_y)
        colchol_packed = pm.LKJCholeskyCov('colcholpacked', n=n_y, eta=2,sd_dist=sd_dist)
        colchol = pm.expand_packed_triangular(n_y, colchol_packed)

        # Concatenate data into a big matrix[[xf tf], [xc tc], [x_pred tf]]
        xf1 = tt.concatenate([xf, tt.fill(tt.zeros([n,q]), tf)], axis = 1)
        x_pred1 = tt.concatenate([x_pred, tt.fill(tt.zeros([n_pred,q]), tf)], axis = 1)
        X = tt.concatenate([xf1, xc, x_pred1], axis = 0)
        # Concate data into a big matrix[[yf], [yc], [y_pred]]
        y = tt.concatenate([yf, yc, y_pred], axis = 0)

        # Covariance funciton of gaussian process
        cov_z = eta1**2 * pm.gp.cov.ExpQuad((p+q), ls=lengthscale)
        # Gaussian process with covariance funciton of cov_z
        gp = MultiMarginal(cov_func = cov_z)

        # Bayesian inference
        matrix_shape = [n+m+n_pred,n_y]
        outcome = gp.marginal_likelihood("outcome", X=X, y=y, colchol=colchol, noise=sigma1, matrix_shape=matrix_shape)
        trace = pm.sample(n_sample,cores=1)  # Put the number of samples you designed to n_sample such as 250 or 500, etc.

    # Data collection and visualization
    pm.summary(trace).to_csv(output_folder + '/trace_summary_{}.csv'.format(case_name))
    pd.DataFrame(np.array(trace['tf'])).to_csv(output_folder + '/tf__{}.csv'.format(case_name))

    name_columns = []
    n_columns = n_pred
    for i in range(n_columns):
        for j in range(n_y):
            name_columns.append('y'+str(j+1)+'_pred'+str(i+1))
    y_prediction = pd.DataFrame(np.array(trace['y_pred']).reshape(n_trace,n_pred*n_y),columns=name_columns)  # Put the number of trace to n_trace based on the above trace results

    # Store predictions
    for i in range(n_y):
        index = list(range(0+i,n_pred*n_y+i,n_y))
        y_prediction1 = pd.DataFrame(y_prediction.iloc[:,index])
        y_prediction1 = y_prediction1*yc_sd[i]+yc_mean[i] # Scale y_prediction back
        y_prediction1.to_csv(output_folder + '/_case_{}_y_pred'.format(case_name)+str(i+1)+'.csv') # Store y_prediction


# Run the model if the file is being executed as the main program
if __name__ == '__main__':
    # Load data
    case_name = 'your_case_name'  # Put your case name  here
    DataComp = np.asarray(pd.read_csv("Path of DataComp + {}".format(case_name)))  # Put the path of simulated data here
    DataField = np.asarray(pd.read_csv("Path of DataField + {}".format(case_name)))[:12,:]  # Put the path of field data here
    DataPred = np.asarray(pd.read_csv("Path of DataField + {}".format(case_name)))
    output_folder = folder

    conf = yaml.load(open("config path + {}".format(case_name)), Loader=yaml.FullLoader)

    # Indicate the number of output
    n_y = len(conf['vc_keys'])+len(conf['yc_keys'])

    print(f'The case you are running is {case_name}')  # Indicate the case you are running

    MultiOutput_Bayesian_Calibration(n_y,DataComp,DataField,DataPred,output_folder)