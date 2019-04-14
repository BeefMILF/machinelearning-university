import torch


__author__ = "Andres FR"


TWOPI = 6.28318530718


def log_gaussian_pdf(x, mu, Sigma):
    """
    Returns the log of the probability density function of a multivariate
    gaussian at the given location, with the given parameters.

    :param torch.FloatTensor x: Location tensor of shape (M,)
    :param torch.FloatTensor mu: Mean tensor of shape (M,)
    :param torch.FloatTensor Sigma: Covariance matrix of shape (M, M)
    :returns: log(gaussian(x, mu, sigma))
    :rtype: torch.float32

    .. doctest::

    >>> x = torch.arange(0, 10).type(torch.float32)
    >>> mu = torch.zeros(10)
    >>> Sigma = torch.eye(10)
    >>> print(log_gaussian_pdf(x, mu, Sigma))
    tensor(-151.6894)
    """
    m, n, mn = x.shape, mu.shape, Sigma.shape
    assert m == n, "x and mu must have same shape!"
    assert len(m) == 1, "Bad x shape!"
    assert len(mn) == 2 and mn[0] == mn[1] == m[0], "Bad Sigma shape!"
    assert torch.allclose(Sigma, Sigma.t()), "Sigma must be symmetric!"
    # If inverse fails, Sigma is not pos. def.
    # (check eigvalsh, try adding epsilon*I for numeric issues)
    Delta = Sigma.inverse()
    determinant = Sigma.det()
    # norm factor:
    fac_term = torch.log(determinant * TWOPI ** m[0]) * -0.5
    # exponent:
    z = x - mu
    exp_term = -0.5 * z.dot(Delta.matmul(z))
    #
    return fac_term + exp_term


def split_tensor_by_indices(tnsr, *indices):
    """
    :param torch.Tensor tnsr: A Tensor of shape (A, ...)
    :param integers indices: a variable number of integers in [0, A-1]
    :returns: A tuple (t1, t2) Where t1 is the subtensor of ``tnsr`` containing
       the given indices, and t2 contains the rest.

    .. doctest::
    >>> split_tensor_by_indices(torch.arange(10, 7, 2, 1)
    (tensor([7, 2, 1]), tensor([0, 3, 4, 5, 6, 8, 9]))
    """
    sh = tnsr.shape
    idx_set = set(indices)
    assert all([0 <= idx < sh[0] for idx in indices]),\
        "Dims must be between zero and len(tnsr)!"
    assert len(indices) == len(idx_set), "no repeated indices allowed!"
    #
    comp_indices = tuple(i for i in range(len(tnsr)) if i not in idx_set)
    return (tnsr[indices, ], tnsr[comp_indices, ])


def split_matrix_by_indices(mat, *indices):
    """
    Analogous to ``split_tensor_by_indices``, splits the given matrix into 4
    submatrices following the given index set.
    :param torch.Tensor mat: A tensor of shape (M, N)
    :returns: A tuple (m11, m12, m21, m22) with the partitioned matrix:
       m11 is a square matrix of shape (len(indices), len(indices)) containing
       the mat[indices, indices] entries. The m22 matrix is its complementary
       square matrix, and m12, m21 contain the entries that combine
       m11 and m22 indexes. E.g. if a 5x5 matrix with indices 3,1 is given:
       - m11 corresponds to ``[[m33, m31], [m13, m11]] (shape AxA)
       - m22 => ``[[m00, m02, m04], [m20, m22, m24], [m40, m42, m44]]``
          (shape BxB)
       - m12 => ``[[m30, m32, m34], [m10, m12, m14]]`` (shape AxB)
       - m21 => ``[[m03, m31], [m23, m21], [m43, m41]]`` (shape BxA)

    .. doctest::

    >>> m = torch.arange(25).view(5,5)
    >>> m11, m12, m21, m22 = split_matrix_by_indices(m, 3, 1)
    >>> print(m, m11, m12, m21, m22, sep="\n")
    """
    m1, m2 = split_tensor_by_indices(mat, *indices)
    m11t, m12t = split_tensor_by_indices(m1.t(), *indices)
    m21t, m22t = split_tensor_by_indices(m2.t(), *indices)
    return m11t.t(), m12t.t(), m21t.t(), m22t.t()


def schur_complement(m11, m12, m21, m22, return_complement_11=False):
    """
    Given the 4 partitions of a matrix, computes the corresponding
    Schur complement.
    :param torch.Tensor m11: A tensor of shape (A, A)
    :param torch.Tensor m12: A tensor of shape (A, B)
    :param torch.Tensor m21: A tensor of shape (B, A)
    :param torch.Tensor m22: A tensor of shape (B, B)
    :param bool return_complement_11: if true, also compute and return m/m11.
    :returns: A tuple with (m/m22, m/m11) if return_complement_11 is true,
       or m/m22 only otherwise.
    .. doctest::

    >>> m = torch.arange(25).view(5,5)
    >>> m11, m12, m21, m22 = split_matrix_by_indices(m, 0, 1)
    >>> schur_m22, schur_m11 = schur_complements(m11, m12, m21, m22, True)
    """
    m_m22 = m11 - m12.matmul(m22.inverse().matmul(m21))
    if return_complement_11:
        m_m11 = m22 - m21.matmul(m11.inverse().matmul(m12))
        return (m_m22, m_m11)
    else:
        return m_m22


def decompose_gaussian(mu, Sigma, dims=[0], gaussian_pdf=log_gaussian_pdf):
    """
    Any multivariate gaussian can be decomposed into 2 gaussians as follows:
    .. math::

       \\mathcal{N}(x, \\mu, S) = \\mathcal{N}(x_2, \\mu_2, S_{22}) \\cdot
       \\mathcal{N}(x_1, \\mu_1 - \frac{S}{S_{22}} \\Delta_{12} (x_2-\\mu_2),
       \\frac{S}{S_{22}})

    Where x1 and x2 form an arbitrary partition of the x vector, and the mu and
    Sigma parameters are partitioned correspondingly. Given the parameters of
    the joint distribution, and the dimensions to be extracted into the x2
    partition, this function returns the two resulting PDFs.

    :param torch.FloatTensor mu: Mean tensor of shape (M,)
    :param torch.FloatTensor Sigma: Covariance matrix of shape (M, M)
    :param list<int> dims: The dimensions that will be part of the x2 partition
    :param function gaussian_pdf: The type of function returned. The decomposed
       parameters will be applied to this function.
    :returns: a tuple with 2 log-PDFs: (gaussian_pdf(x2), gaussian_pdf(x1|x2))
       where the functions have the decomposed parametrization

    .. doctest::
    >>> D, INDICES = 10, [6,4,7,2]
    >>> mu = torch.arange(D).type(torch.float64)
    >>> Sigma = torch.arange(D**2).type(torch.float64).view(D, D)
    >>> Sigma = Sigma.matmul(Sigma.t()) //100 +\
                torch.eye(D).type(torch.float64)*3
    >>> gaussian_2, gaussian_12 = decompose_gaussian(mu, Sigma, INDICES,
                                                     log_gaussian_pdf)
    >>> x = torch.rand(D).sub(0.5).mul(50).type(torch.float64)
    >>> x1, x2 = split_tensor_by_indices(x, *INDICES)
    >>> print(log_gaussian_pdf(x, mu, Sigma),
                               gaussian_2(x2) + gaussian_12(x1, x2))
    """
    m, mm = mu.shape, Sigma.shape
    assert len(m) == 1, "Bad mu shape!"
    assert len(mm) == 2 and mm[0] == mm[1]  == m[0], "Bad Sigma shape!"
    assert (Sigma == Sigma.t()).all(), "Sigma must be symmetric!"
    #
    mu1, mu2 = split_tensor_by_indices(mu, *dims)
    s11, s12, s21, s22 = split_matrix_by_indices(Sigma, *dims)
    d11, d12, d21, d22 = split_matrix_by_indices(Sigma.inverse(), *dims)
    # schur_s22 is also the inverse of d11
    schur_s22 = s11 - s12.matmul(s22.inverse().matmul(s21))
    # The f(x2) component is straightforward: simply a truncation
    def f2(x2):
        return gaussian_pdf(x2, mu2, s22)
    # The f(x1 | x2) component requires to precompute the mean shift:
    mu_shift = mu1 + schur_s22.matmul(d12).matmul(mu2)
    x2_shift = schur_s22.matmul(d12)
    def f1_given_2(x1, x2):
        return gaussian_pdf(x1, mu_shift - x2_shift.matmul(x2), schur_s22)
    #
    return f2, f1_given_2
