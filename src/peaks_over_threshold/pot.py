from __future__ import annotations

import math

import numpy as np
import numba as nb
from scipy.optimize import minimize


class SPOT:
    def __init__(self, q: float = 1e-3) -> None:
        self.q = q
        self.calibrated = False
        self.t = 0
        self.zq = 0

        self.n = 0
        self.excesses = np.zeros(1, dtype=float)

    def fit(self, X: np.ndarray, init_quantile: float = 0.98):
        """
        Calibrate the algorithm.
        """
        zq, t = pot(X, self.q, init_quantile=init_quantile)
        self.t = t
        self.zq = zq
        self.calibrated = True

        self.excesses = X[X > t] - t
        self.n = X.size

        return self

    def fit_predict(
        self,
        X: np.ndarray,
        *,
        num_inits: int | None = None,
        init_quantile: float = 0.98,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        SPOT.

        Parameters:
            X: 1d time series to find extreme values in.
            num_inits: number of elements to be used for calibrating the zq and t.
            init_quantile: quantile for estimating.

        Returns: a tuple of two arrays with the same size as X
            * The thresholds zq;
            * The extremes indicator array;
        """
        thresholds = np.zeros_like(X)
        extremes = np.zeros_like(X, dtype=bool)

        if not self.calibrated:
            assert (
                num_inits is not None
            ), "Requiring number of elements for calibrating the algorithm."

            # Calibrate the algorithm.
            self.fit(X[:num_inits], init_quantile=init_quantile)

            # Set the initial thresholds.
            thresholds[:num_inits] = self.zq

            start_predict_idx = num_inits
        else:
            start_predict_idx = 0

        # Predict.
        for i in range(start_predict_idx, X.size):
            value = X[i]

            # Found an extreme value.
            if value > self.zq:
                extremes[i] = True
            else:
                self.n += 1

                if value > self.t:
                    # Reestimate EVD's parameters and update the zq.
                    self.excesses = np.append(self.excesses, value - self.t)

                    gamma, sigma = _grimshaw(self.excesses)
                    self.zq = _calculate_threshold(
                        q=self.q,
                        gamma=gamma,
                        sigma=sigma,
                        n=self.n,
                        Nt=self.excesses.size,
                        t=self.t,
                    )

            # Update the threshold.
            thresholds[i] = self.zq

        return thresholds, extremes


def pot(
    X: np.ndarray,
    q: float = 1e-3,
    *,
    init_quantile: float = 0.98,
) -> tuple[float, float]:
    """
    Estimate the threshold zq such that P(X > zq) < q,
    the Algorithm 1 in the paper.

    Parameters:
        X: 1d time-series to estimate the z_q.
        q: the probability of the extreme values occur.
        init_quantile: the quantile for estimating.

    Returns:
        a tuple of zq and t.

    Raises: ValueError if the number of excesses is 0.
    """
    t = np.quantile(X, init_quantile)

    excesses = X[X > t] - t
    if excesses.size == 0:
        raise ValueError("Cannot estimate the threshold zq.")

    gamma, sigma = _grimshaw(excesses)

    zq = _calculate_threshold(
        q=q, gamma=gamma, sigma=sigma, n=X.size, Nt=excesses.size, t=t
    )

    return zq, t


def _grimshaw(
    excesses: np.ndarray,
    *,
    num_candidates: int = 10,
    epsilon: float = 1e-8,
) -> tuple[float, float]:
    ymin, ymax, ymean = (
        excesses.min(),
        excesses.max(),
        excesses.mean(),
    )

    # Calculate the bounds for root search.
    a = -1 / ymax
    b = 2 * (ymean - ymin) / (ymean * ymin)
    c = 2 * (ymean - ymin) / (ymin * ymin)

    # Instead of using root search, we perform a minimization problem.
    candidates_1 = minimize(_obj, args=excesses, bounds=(a + epsilon, -epsilon))
    candidates_2 = minimize(_obj, args=excesses, bounds=(b, c))
    candidates = np.concatenate([candidates_1, candidates_2])

    # Calculate gamma and sigma.
    gamma = _v(excesses, candidates) - 1
    sigma = gamma / candidates

    # TODO: include gamma and sigma for candidate = 0.

    # Calculate the log-likelihood and choose the best one.
    best_idx = np.argmax(_log_likelihood(excesses, gamma, sigma))

    return gamma[best_idx], sigma[best_idx]


@nb.guvectorize(
    [nb.float32[:], nb.float32, nb.float32, nb.float32[:]],
    "(n),(),()->()",
    nopython=True,
)
def _log_likelihood(excesses: np.ndarray, gamma: float, sigma: float, res: np.ndarray):
    Nt = excesses.size

    # TODO: what if sigma and gamma is zero or negative?
    x = gamma / sigma
    res[0] = -Nt * np.log(sigma) - (1.0 + 1.0 / gamma) * np.sum(
        np.log(1.0 + x * excesses)
    )


@nb.njit()
def _obj_deriv(x: np.ndarray, excesses: np.ndarray):
    pass


@nb.njit()
def _obj(x: np.ndarray, excesses: np.ndarray):
    # TODO: calculate the derivative here to save some computations.

    w = _u(excesses, x) * _v(excesses, x) - 1
    return np.sum(w**2)


@nb.njit()
def _u(excesses: np.ndarray, x: np.ndarray):
    return np.mean(1.0 / (1.0 + x[None, ...] * excesses[..., None]), axis=-1)


@nb.njit()
def _v(excesses: np.ndarray, x: np.ndarray):
    # TODO: support numpy broadcasting.
    return 1.0 + np.mean(np.log(1 + x[None, ...] * excesses[..., None]), axis=-1)


@nb.njit()
def _calculate_threshold(
    *, q: float, gamma: float, sigma: float, n: int, Nt: int, t: float
) -> float:
    tau = q * n / Nt

    if math.isclose(gamma, 0):
        # TODO
        return t - sigma * math.log(tau)
    else:
        return t + (sigma / gamma) * (pow(tau, -gamma) - 1)
