from dataclasses import dataclass

import numpy as np
import properscoring as ps
from numpy.typing import NDArray
from scipy import stats


@dataclass
class CRPSSubDecomposition:
    crps: float
    cs: NDArray[np.float64]
    alphas: NDArray[np.float64]
    betas: NDArray[np.float64]
    ps: NDArray[np.float64]


@dataclass
class CRPSDecomposition:
    crps_arr: NDArray[np.float64]
    potential_crps_arr: NDArray[np.float64]
    reliability_arr: NDArray[np.float64]
    uncertainty: np.float64

    def aggregate(self) -> "CRPSDecomposition":
        self.crps: np.float64 = np.nansum(self.crps_arr)
        self.reliability: np.float64 = np.nansum(self.reliability_arr)
        self.potential_crps: np.float64 = np.nansum(self.potential_crps_arr)
        self.resolution: np.float64 = self.uncertainty - self.potential_crps
        return self


def crps(y: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Continuous Rank Probability Score.

    Args:
        y: Observations. np.ndarray, shape: [M, 1]
        q: Predicted Distributions (samples). np.ndarray, shape: [M, N]
    """
    crps_array: np.ndarray = ps.crps_ensemble(y[:, 0], q)  # type: ignore
    return crps_array


def _crps_sub_decomposition(y: float, q: NDArray[np.float64]) -> CRPSSubDecomposition:
    """
    Continuous rank probability score decomposition helper. Compute alphas, betas, probabilities,
    and cs needed in full decomposition.

    Args:
        - y: float. Observation.
        - q: np.ndarray, shape (M, 1). Note, q must be sorted in ascending order.

    See section 4 (equations 26 and 27) in:
    "Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems"
    https://journals.ametsoc.org/view/journals/wefo/15/5/1520-0434_2000_015_0559_dotcrp_2_0_co_2.xml

    Note: In the associated paper, y is x_a and q is x.
    """

    N = q.shape[0]
    cs = []
    ps = []
    alphas = []
    betas = []

    # Equation 27
    if y < q[0]:
        alpha = 0.0
        beta = q[0] - y
        pi = 0.0
    else:
        alpha = 0.0
        beta = 0.0
        pi = 0.0

    c = (alpha * pi**2) + beta * (1 - pi) ** 2
    alphas.append(alpha)
    betas.append(beta)
    ps.append(pi)
    cs.append(c)

    # Equation 26
    for i in range(1, N):
        i -= 1  # Decrement i to ensure indexing matches that of paper eq. 26
        pi = (i + 1) / N

        if y >= q[i + 1]:
            alpha = q[i + 1] - q[i]
            beta = 0.0

        elif (q[i + 1] > y) and (y >= q[i]):
            alpha = y - q[i]
            beta = q[i + 1] - y

        elif y < q[i]:
            alpha = 0.0
            beta = q[i + 1] - q[i]

        c = (alpha * pi**2) + beta * (1 - pi) ** 2
        alphas.append(alpha)
        betas.append(beta)
        ps.append(pi)
        cs.append(c)

    # Equation 27
    if q[N - 1] < y:
        alpha = y - q[N - 1]
        beta = 0.0
        pi = 1.0
    else:
        alpha = 0.0
        beta = 0.0
        pi = 1.0

    c = (alpha * pi**2) + beta * (1 - pi) ** 2
    alphas.append(alpha)
    betas.append(beta)
    ps.append(pi)
    cs.append(c)

    return CRPSSubDecomposition(
        crps=np.array(cs).sum(),
        cs=np.array(cs),
        alphas=np.array(alphas),
        betas=np.array(betas),
        ps=np.array(ps),
    )


def crps_decomposition(y: NDArray[np.float64], q: NDArray[np.float64]) -> CRPSDecomposition:
    """
    Continuous Rank Probability Score Decomposition across M examples.

    Args:
        y: np.ndarray, shape: [M, 1]
        q: np.ndarray, shape: [M, N]

    Where M = number of examples (observation - predicted distributions pairs) and N =
    number of samples in a given predicted distribution.

    See section 4 (equations 30, 31, 35):
    "Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems"
    https://journals.ametsoc.org/view/journals/wefo/15/5/1520-0434_2000_015_0559_dotcrp_2_0_co_2.xml
    """

    q.sort(axis=1)

    alpha_list = []
    beta_list = []
    p_list = []
    for i in range(q.shape[0]):
        crps_sub_decomp = _crps_sub_decomposition(y[i][0], q[i])
        alpha_list.append(crps_sub_decomp.alphas)
        beta_list.append(crps_sub_decomp.betas)
        p_list.append(crps_sub_decomp.ps)

    alpha_arr = np.array(alpha_list)
    beta_arr = np.array(beta_list)
    p_arr = np.array(p_list)[0]

    alpha_means = alpha_arr.mean(axis=0)
    beta_means = beta_arr.mean(axis=0)
    g_means = alpha_means + beta_means
    o_means = beta_means / (alpha_means + beta_means)

    reliability_arr = g_means * (o_means - p_arr) ** 2
    potential_crps_arr = g_means * o_means * (1 - o_means)
    crps_arr = reliability_arr + potential_crps_arr
    uncertainty_ = uncertainty(y[:, 0])

    return CRPSDecomposition(
        crps_arr=crps_arr,
        reliability_arr=reliability_arr,
        potential_crps_arr=potential_crps_arr,
        uncertainty=uncertainty_,
    ).aggregate()


def uncertainty(y: NDArray[np.float64], step: float = 0.001) -> np.float64:
    """
    See equation 12 in:
    "Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems"
    https://journals.ametsoc.org/view/journals/wefo/15/5/1520-0434_2000_015_0559_dotcrp_2_0_co_2.xml

    Also, for a non vectorized version, see equation 19 of the above paper.
    """
    yaxis = np.arange(y.min() - 1, y.max() + 1, step)
    y_ecdf_result = stats.ecdf(y)
    y_ecdf = y_ecdf_result.cdf.evaluate(yaxis)
    unc_: np.float64 = (y_ecdf * (1 - y_ecdf) * step).sum()
    return unc_


def crps_skill_score(
    y: NDArray[np.float64], q: NDArray[np.float64], y_ref: NDArray[np.float64]
) -> float:
    """
    Continuous Rank Probability Skill Score.
    "The Discrete Brier and Ranked Probability Skill Scores"
    https://journals.ametsoc.org/view/journals/mwre/135/1/mwr3280.1.xml

    Args:
        y: np.ndarray, shape: [M, 1]
        q: np.ndarray, shape: [M, N]
        y_ref: np.ndarray, shape: [K, 1] (K = length of historical observations)

    See here for R implementation: https://search.r-project.org/CRAN/refmans/s2dv/html/CRPSS.html
    """
    actual_crps = crps(y, q).mean()
    y_ref = np.repeat(y_ref.T, repeats=y.shape[0], axis=0)
    marginal_crps = crps(y, y_ref).mean()
    crps_ss: float = 1.0 - (actual_crps / marginal_crps)
    return crps_ss