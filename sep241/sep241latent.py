import numpy as np
import pymc3 as pm
from pymc3.gp.util import infer_shape
import theano.sparse as ts
from sep241covariance import SparseCov
from scipy.sparse.linalg import spsolve, splu
from scipy.linalg import LinAlgError
from scipy.sparse import diags
from sksparse.cholmod import cholesky


class SparseLatent:
    R"""
    This class is like pymc3.gp.Latent, except it uses a sparse covariance matrix to save time
    and memory and only supports prior. cov_func must be an instance of SparseCov.
    See documentation of SparseCov for use conditions.

    :param mean_func: The mean function.
    :type mean_func: pymc3.gp.mean.Mean
    :param cov_func: The covariance function. See SparseCov.
    :type cov_func: SparseCov
    """

    def __init__(self, mean_func, cov_func):
        if not isinstance(cov_func, SparseCov):
            raise ValueError('Covariance function must be wrapped by SparseCov.')
        self.mean_func = mean_func
        self.cov_func = cov_func

    def _format_initval(self, initval, L):
        if initval is None:
            return None
        return spsolve(L, initval, 'NATURAL').reshape(-1, 1)

    def _parametrization(self, X, jitter='auto', initval=None):
        size = infer_shape(X)
        mu = self.mean_func(X)
        cov = self.cov_func(X)

        if jitter == 'auto':
            jitter = self.cov_func.auto_jitter(cov)
        factor = cholesky(cov, beta=jitter, mode='simplicial', ordering_method='natural')
        L = factor.L()
        initval = self._format_initval(initval, L)
        return size, mu, L, initval

    def _build_prior(self, name, X, reparameterize=True, jitter='auto', initval=None):
        if reparameterize:
            size, mu, L, initval = self._parametrization(X, jitter=jitter, initval=initval)
            v = pm.Normal(name + '_rotated_', 0, 1, shape=(size, 1),
                          testval=initval)
            f = pm.Deterministic(name, mu + ts.structured_dot(L, v).flatten())
        else:
            size = infer_shape(X)
            mu = self.mean_func(X)
            cov = self.cov_func(X)
            f = pm.Normal(name, 0, 1, shape=size, testval=initval)
        return f

    def prior(self, name, X, reparameterize=True, jitter='auto', initval=None):
        R"""
        Returns the GP prior distribution evaluated over the input
        locations `X`.
        This is the prior probability over the space
        of functions described by its mean and covariance function.

        :param name: Name of the random variable.
        :type name: string
        :param X: Function input values.
        :type X: array-like
        :param reparameterize: Re-parameterize the distribution by rotating the random
            variable by the Cholesky factor of the covariance matrix.
        :type reparameterize: bool
        :param jitter: A correction added to the diagonal of the covariance matrix for numerical
            stability. If set to 'auto', computes a minimum to ensure the covariance
            matrix is positive semi-definite.
            Defaults to 'auto'.
        :type jitter: scalar or 'auto'
        :param initval: Initial value for rotated distribution. Defaults to None.
        :type initval: array-like, optional
        """
        f = self._build_prior(name, X, reparameterize, jitter, initval=None)
        self.X = X
        self.f = f
        return f
