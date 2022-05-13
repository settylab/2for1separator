from scipy.sparse import csc_matrix
import numpy as np
import theano
import theano.tensor as at
from pymc3.gp.util import infer_shape
import warnings

THRESHOLD_DEFAULT = 1e-8

class SparseCov:
    R"""
    A callable wrapper class for covariance functions that represents the covariance matrix
    sparsely under certain assumptions:
    
    The covariance function k(x, y) must have a positive range, only dependent on the 
    distance d = abs(x - y), and be strictly decreasing with respect to d. 
    Further, the data must be strictly increasing, 1-dimensional, and contain integers
    (divisible by 1, not necessarily type int). Since the data is strictly increasing, for
    any data point x, as y becomes farther away in terms of index, d must be increasing.
    Since k(x, y) is strictly decreasing with respect to d, for any data point x, once we
    find a y for which k(x, y) <= threshold, every data point beyond y must also satisfy
    k(x, y) <= threshold. 
    
    So if we choose a threshold at which we consider the covariance
    negligible, the significant data will be clustered around the diagonal. So we can compute
    the covariance matrix in O(nm) time and memory, where n is the length of the data and m
    is the average number of points y satisfying k(x, y) <= threshold across all x.

    :param cov_func: A pymc3 covariance function. Must have a positive range and be strictly
        decreasing with respect to the distance between points.
    :type cov_func: pymc3.gp.cov.Covariance
    :param threshold: Do not calculate or represent any covariances k <= threshold.
        Defaults to 1e-8.
    :type threshold: float, optional
    """

    def __init__(self, cov_func, threshold=THRESHOLD_DEFAULT):
        if threshold < 0:
            raise ValueError('Threshold must be positive.')

        a = at.dvector()
        a.tag.test_value = np.ones(2)

        # We can think of the covariance function k(x, y) as a monotonically decreasing function of the
        # distance between x and y, notated k(d). Here the covariance function is structured so
        # it transforms a vector of distances.
        with warnings.catch_warnings():
            # pymc covariance functions interpret symbolic matrices as having the wrong input
            # dimensions, throwing an unnecesssary warning.
            warnings.simplefilter("ignore")
            self.cov_func = theano.function([a], cov_func(np.array([[0.0]]), a.reshape((-1, 1))).flatten())


        self.threshold = threshold
        
        # Since k is monotonically decreasing, k(d) <= threshold implies k(d + c) <= threshold for any
        # positive constant c. We assume integer data, so we only really need the first integer d for
        # which k(d) <= threshold.
        self.inv = np.ceil(self._find_inverse())

    def _find_inverse(self, upper=1e+15):
        R"""
        Finds the minimum distance d such that the k(d) <= threshold for all distances equal or greater.
        Searches between 0 and upper.
        Parameters
        ----------
        upper: float
            Assume that the inverse is less than this value. Must be greater than 0. Defaults to 1e+15.
        """

        lower = 0.0

        # Invariant:
        # k(lower) > threshold
        # k(upper) <= threshold
        # lower < upper
        while upper > np.nextafter(lower, np.inf):
            # If we are in the loop, there is at least one representable floating point
            # between upper and lower.

            middle = (upper / 2.0) + (lower / 2.0)
            
            # In case of numerical issues, make sure lower < middle < upper. Because there is a
            # floating point between upper and lower, we can make this inequality exclusive.
            if middle <= lower:
                middle = np.nextafter(lower, np.inf)
            elif middle >= upper:
                middle = np.nextafter(upper, -np.inf)

            sim = self.cov_func(np.array([middle]))
            if sim > self.threshold:
                # k(middle) > threshold.
                lower = middle
            else:
                # k(middle) <= threshold
                upper = middle
            # Since we had lower < middle < upper, and then we set one of lower or upper to middle,
            # when the loop repeats or terminates we have lower < upper.

        # When the loop terminates, we have:
        # k(lower) > threshold
        # k(upper) <= threshold
        # lower < upper
        # upper <= np.nextafter(lower, np.inf)

        # Simplifying the last two statements, lower < upper <= np.nextafter(lower, np.inf), so
        # upper = np.nextafter(lower, np.inf). So upper is the first distance d for which
        # k(d) <= threshold and lower is the last distance d for which k(d) > threshold.
        return upper

    def _find_spine(self, X):
        R"""
        For each datapoint in X, finds the number of the number of datapoints with distance d such that
        k(d) > threshold. Returns an array for the number of datapoints to the left within this distance
        and an array for the number of datapoints to the right within this distance.
        Parameters
        ----------
        X: array-like
            Function input values.
        """
        n = infer_shape(X)
        
        # Side left to not include the exact threshold.
        rightspine = np.searchsorted(X, X + self.inv, side='left') - np.arange(1, n + 1)
        reverse = -np.flip(X)
        leftspine = np.flip(np.searchsorted(reverse, reverse + self.inv, side='left') - np.arange(1, n + 1))
        return leftspine, rightspine

    def _align(self, X):
        R"""
        Finds all the pairs of entries in X with distance d such that k(d) > threshold. Returns a tuple containing
        the data, indices and indptr that place the distance between x[i] and x[j] in locations (i, j) and (j, i)
        in a csc matrix.
        Parameters
        ----------
        X: array-like
            Function input values.
        """
        X = X.flatten().astype(np.int64)
        n = infer_shape(X)

        # Number of dense entries left of the diagonal for each row, number of entries right of the diagonal for each row
        nnzleft, nnzright = self._find_spine(X)

        # Total number of dense entries for each row
        nnzrow = nnzleft + nnzright + 1

        # Number of entries preceding each row
        indptr = np.append(0, np.cumsum(nnzrow))

        # Imagine assigning indices one a at time. When assigning an index to an entry, the next entry in the same row
        # is one index greater.
        differences = np.ones(np.sum(nnzrow), dtype=np.int64)

        # When assigning the index of the first entry of a row, move left one square for each entry right of the
        # diagonal in the previous row and one square for each entry left of the diagonal in this row. Still move
        # right one index because the diagonal shifts one index. Also subtract 1 from the first entry of differences
        # to start at index 0.
        differences[indptr[:-1]] -= np.append(1, nnzleft[1:] + nnzright[:-1])
        indices = np.cumsum(differences, out=differences)

        # A_ij = |X[i] - X[j]|
        data = np.abs(np.repeat(X, nnzrow) - X[indices])
        return data, indices, indptr

    def auto_jitter(self, matrix):
        R"""
        Calculates the jitter necessary to guarantee positive definiteness of the
        sparsified covariance matrix. Assumes that the matrix would be positive
        semi-definite before the sparsification.

        :param matrix: The NxN sparsified covariance matrix.
        :type matrix: scipy.sparse.spmatrix
        :return: jitter to guarantee positive definiteness.
        :rtype: float
        """
        # A Hermitian strictly diagonally dominant matrix with non-negative diagonals
        # is PD. Source: https://en.wikipedia.org/wiki/Diagonally_dominant_matrix.
        # Consider the sparsification matrix. It has zeros wherever we leave
        # the original matrix alone and negative values where we set the original
        # matrix to zero, which can only be in off diagonal entries. If we set the
        # diagonal of the sparsification matrix greater than the magnitude of the
        # sum of corresponding rows, the sparsification matrix is real, strictly 
        # diagonally dominant, and has non-negative diagonals, and is therefore PD.
        # Since the original covariance matrix is PSD and the sparsification matrix
        # is PD, their sum is PD.

        n = infer_shape(matrix)

        # rows and cols order the indices of the non-zero entries of the matrix
        # left to right of row 0, left to right of row 1, and so on.
        rows, cols = matrix.nonzero()

        # We want to find the number of zeros to the left of the leftmost non-zero entry and
        # to the right of the rightmost entry in each row. We can assume that every row has
        # at least one element because the diagonal is assumed to be non-zero. The ith index
        # of np.diff(rows) == 1 is True iff the the row changes between the ith entry and the
        # (i+1)th entry. So np.diff(rows) == 1 is a mask for the entries before each row change,
        # i.e. the last nonzero entry of each row except the last row, since there is no row
        # change from the last row to a nonexistent row.

        # The first nonzero entry of each row occurs one entry after the last nonzero entry
        # of the previous row. Since np.diff(rows) == 1 is a mask for the last entries of
        # each row except the last, by padding the mask with a True in the beginning, we
        # offset the mask by one, so the new mask accesses the first entries of each row,
        # including the first non-zero entry of row 0, which must be in row 0 column 0. The
        # column indices of the first non-zero entries of each row is the same as the number
        # of zero entries on the left side. 
        leftspine = cols[np.append(True, np.diff(rows) == 1)]

        # By padding the mask with a True at the end, the new mask accesses the last nonzero
        # entries of each row, including the last non-zero entry of the last row, which must
        # be in row n - 1 column n - 1. The column indices of the last non-zero entries of
        # each row give the number of zero or non-zero entries to the left of the last non-zero
        # entries of each row, not including the entry itself, so if i is the column index
        # of the last non-zero entry for a row, the number of zero entries to the right of the
        # last non-zero entry is n - i - 1.
        rightspine = n - (cols[np.append(np.diff(rows) == 1, True)] + 1)

        # The inverse is defined as the minimum distance that the covariance function
        # transforms to a value less than or equal to the threshold. Distances that get
        # thrown out must be at least the inverse. Our strategy is to assume the worst:
        # that the distances increase as little as possible. The largest distance that we
        # may have to assume is for the case that the first or last row has n - 1 zeros.
        # In this case the largest distance is self.inv + n - 2.
        distances = np.arange(self.inv, self.inv + n - 1)

        # The associated covariances for all the possible distances.
        costs = self.cov_func(distances)
        costs = np.append(0, costs)

        # A series of m negatives cannot have a greater cost than
        # k(inv) + ... + k(inv + m - 1). So
        # seriescost[m] is the maximum cost of a series of m negatives.
        seriescosts = np.cumsum(costs)

        # Rowcosts gives the maximum amount subtracted in each row by the sparsification.
        rowcosts = seriescosts[leftspine] + seriescosts[rightspine]

        # For consistent variance, choose the same jitter for the whole diagonal.
        return np.max(rowcosts) + 1e-6

    def __call__(self, X):
        R"""
        Returns a sparse covariance matrix, only calculating entries greater than the threshold.
        Because the input data must be strictly increasing, all dense entries of the resulting
        covariance matrix will be located around the main diagonal.
        Parameters
        ----------
        X: array-like
            Function input values.
        """
        # Get the distance of all data points to compare
        data, indices, indptr = self._align(X)

        # Transform the distances with the covariance function
        data = self.cov_func(data)

        return csc_matrix((data, indices, indptr))
