"""The following code performs measurement error correction via Iterative Bayesian
Unfolding and uncertainity estimation with a Monte Carlo Simulation.

Details about the theory behind the method can be found in Appendix A of Matteo
Pompili's PhD Thesis [1], and in references [2-4].

Matteo Pompili, August 2021.

References:
[1] Pompili, Matteo. "Multi-Node Quantum Networks with Diamond Qubits."
    PhD thesis, Delft University of Technology, 2021.
[2] D'Agostini, Giulio. "A multidimensional unfolding method based on Bayes' theorem."
    Meth. in Phys. Res. A362 (1995) 487
[3] D'Agostini, Giulio. "Improved iterative Bayesian unfolding."
    arXiv preprint arXiv:1010.0632 (2010).
[4] Nachman, Benjamin, et al. "Unfolding quantum computer readout noise."
    npj Quantum Information 6.1 (2020): 1-7.
"""

from typing import List, Tuple

import numpy as np
from numba import jit

# pylint: disable=invalid-name


@jit(nopython=True, cache=True)
def bayesian_unfolding(m: np.ndarray, R: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Perform one bayesian unfolding step, producing an updated distribution of
    measurement outcomes (posterior) given the measured event outcomes `m`, the
    transition matrix `R` and the `prior` distribution of causes.

    Arguments:
        m[i]: number of times event `i` has happened;
        R[i,j]: prob(event=i | cause=j);
        prior[j]: prob(cause=j);
    Return:
        posterior[j]: number of times event `j` has happened after the correction step
    """
    posterior = np.zeros_like(prior)
    denominators = np.dot(R, prior)
    for j in range(prior.size):
        for i in range(prior.size):
            if denominators[i] != 0:
                # If one the causes has probability 0, skip it.
                posterior[j] += R[i, j] * m[i] / denominators[i]
        posterior[j] *= prior[j]
    return posterior


@jit(nopython=True, cache=True)
def iterative_bayesian_unfolding(
    m: np.ndarray, R: np.ndarray, niter_max: int, break_tol: float
) -> np.ndarray:
    """Perform the iterative bayesian unfolding to correct for known measurement error
    R. The iteration is completed either after `niter_max` steps, or when the posterior
    probability distribution changes less than `break_tol` in between steps. Keep in
    mind that the tolarance is calculated on the number of events, an integer, so the
    default one works in most cases.

    Arguments:
        m[i]: number of times event `i` has happened;
        R[i,j]: prob(event=i | cause=j);
        niter_max: maximum number of iterations;
        break_tol: tolerance used to break the iteration, set to 0 to disable tolerance
            breaking and always use niter_max;
    Return:
        posterior[j]: number of times event `j` has happened after the unfolding
    """

    prior = np.ones_like(m) * np.sum(m) / m.size
    break_tol_total = break_tol * prior.size
    tol_total = break_tol_total + 1
    niter = 0
    while (niter <= niter_max) and (tol_total > break_tol_total):
        posterior = bayesian_unfolding(m, R, prior)
        tol_total = np.sum(np.abs(prior - posterior))
        prior = posterior
        niter += 1

    return posterior


def generate_R_MC(R_N: List[np.ndarray], N_MC: int) -> np.ndarray:
    """We generate N Monte-Carlo Readout Error matrices (R) using the Dirichlet
    distribution. The Dirichlet distribution generates Multinomial probability
    distributions based on the outcomes of a measurement. Based on
    alpha_prior = [1, 1, ...., 1], which produces a flat Multinomial probability
    distribution, and the measured number of outcomes per state R_m = [m0, m1, ...] in
    the SSRO calibration, we can update the probability distribution
    alpha_posterior = [1 + m0, 1 + m1]. This posterior is then used to generate N
    Monte-Carlo Readout Error matrices that re-sample the measured one and allow for
    uncertainity propagation.
    """

    R_MC_qbit = np.zeros((N_MC, len(R_N), 2, 2))

    for qbit, R_m in enumerate(R_N):
        R_MC_qbit[:, qbit, :, 0] = np.random.dirichlet(alpha=R_m[:, 0] + 1, size=N_MC)
        R_MC_qbit[:, qbit, :, 1] = np.random.dirichlet(alpha=R_m[:, 1] + 1, size=N_MC)

    for qbit in range(len(R_N)):
        if qbit == 0:
            kron_product = R_MC_qbit[:, 0]
        else:
            kron_product = np.einsum(
                "nab,ncd->nacbd", kron_product, R_MC_qbit[:, qbit]
            ).reshape((-1, 2 << qbit, 2 << qbit))

    return kron_product


def MC_covariance_estimation(
    m: np.ndarray, R_MC: np.ndarray, p: np.ndarray
) -> np.ndarray:
    """Perform the Monte-Carlo simulation to estimate the covariance matrix of the
    populations.

    Arguments:
        m[i]: number of times event `i` has happened;
        R_MC: a list of Monte-Carlo generated sxs readout error matrices, it should be
            the result of generate_R_MC();
        p: final unfolded probability distribution, of which we will find the covariance
            matrix;
    Returns:
        cov: an sxs covariance matrix, where cov[i,j] is Cov[p_i, p_j].
    """

    N_MC = np.shape(R_MC)[0]

    U_MC_numerator = np.einsum("nij, j -> nji", R_MC, p)
    U_MC_denominator = 1.0 / np.einsum("nik, k -> ni", R_MC, p)
    U_MC = np.einsum("nji, ni-> nji", U_MC_numerator, U_MC_denominator)

    m_MC = np.random.dirichlet(alpha=m + 1, size=N_MC)
    p_MC = np.einsum("ni,nji->nj", m_MC, U_MC)

    # Now we calculate the statistical quantities
    E_p = np.mean(p_MC, axis=0)
    E_pij = np.einsum("ni, nj -> ij", p_MC, p_MC) / N_MC
    cov = E_pij - np.einsum("i, j -> ij", E_p, E_p)
    return cov


def ssro_correction(
    m_list: List[np.ndarray],
    R_N: List[np.ndarray],
    N_MC=250_000,
    niter_max=1000,
    break_tol=1e-3,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Perform the Single Shot Readout (SSRO) error correction, or measurement error
    correction, on a list of measurements. It uses Iterative Bayesian Unfolding for
    error correction and a Monte-Carlo simulation for uncertainty estimation. See
    Appendix A of Matteo Pompili's PhD Thesis for details.

    Arguments:
        m_list: list of measured histograms (for example different sweep points). In
            case of a single measurement, pass a list with a single array in it;
        R_N: readout error histogram as measured by the SSRO calibration. Each column
            should be an histogram, e.g. R_N[i, j] = number of times state `i` was
            measured when state `j` was prepared;
        N_MC: number of Monte-Carlo resampling to perform. Processign time (and memory
            required) scales approximately linearly with this. Higher number reduces
            fluctuations in estimated uncertainties and covariances;
        niter_max: number of maximum iterations to perform in the iterative Bayesian
            unfolding step;
        break_tol: if two successive Bayesian unfolding steps differ less than this,
            stop the unfolding.

    Returns:
        p_list: list of corrected (unfolded) measurement probability distributions;
        unc_list: list of uncertainties for p_list;
        cov_list: list of covariance matrices for p_list.
    """

    # Normalize readout matrices and tensor-product them together,
    # since we assume the readout errors are independent from each other.
    R_tot = 1
    for R in R_N:
        R_tot = np.kron(R_tot, R / R.sum(axis=0))

    # Using the Iterative Bayesian Unfolding, calculate the corrected number of events and
    # normalize them to obtain corrected probabilties distributions.
    unfolded_events = [
        iterative_bayesian_unfolding(m, R_tot, niter_max, break_tol) for m in m_list
    ]
    p_list = [u / u.sum() for u in unfolded_events]

    # To calculate uncertainties, generate N_MC resampled readout error matrices.
    # We use the same R_MC ensemble for all the sweep points, instead of generating
    # a new one for each sweep point.
    R_MC = generate_R_MC(R_N, N_MC)

    # For each sweep point, calculate the covariance matrix and from it the uncertainties.
    cov_list = []
    unc_list = []
    for m, p in zip(m_list, p_list):
        cov = MC_covariance_estimation(m, R_MC, p)
        cov_list.append(cov)
        unc_list.append(np.sqrt(cov.diagonal()))
    return p_list, unc_list, cov_list
