from __future__ import annotations

from math import log

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


class DSPOT:
    def __init__(self, depth: int, q: float = 1e-3) -> None:
        self.q = q
        self.calibrated = False
        self.t = 0
        self.zq = 0
        self.depth = depth

        self.n = 0
        self.excesses = np.zeros(1, dtype=float)
        self.window = np.zeros(1, dtype=float)
        self.window_avg = 0

    def fit(self, X: np.ndarray, init_quantile: float = 0.98):
        depth = self.depth
        assert X.size > depth, "Time series too short to calibrate the algorithm"

        window_avg = np.mean(X[:depth])

        remains = X[depth:]
        diff = np.zeros_like(remains)

        for i in range(remains.size):
            diff[i] = remains[i] - window_avg

            # Update the window avg.
            window_avg += (X[i] - X[i - depth]) / depth

        # Calculate the zq and t.
        zq, t = pot(diff, self.q, init_quantile=init_quantile)
        self.t = t
        self.zq = zq
        self.calibrated = True

        self.excesses = diff[diff > t] - t
        self.n = X.size

        self.window = X[-depth:]
        self.window_avg = window_avg

        return self

    def fit_predict(
        self,
        X: np.ndarray,
        *,
        num_inits: int | None = None,
        init_quantile: float = 0.98,
    ) -> tuple[np.ndarray, np.ndarray]:
        thresholds = np.zeros_like(X)
        extremes = np.zeros_like(X, dtype=bool)

        if not self.calibrated:
            assert (
                num_inits is not None
            ), "Requiring number of elements for calibrating the algorithm."

            # Calibrate the algorithm.
            start_predict_idx = num_inits + self.depth
            self.fit(X[:start_predict_idx], init_quantile=init_quantile)

            # Set the initial thresholds.
            thresholds[:start_predict_idx] = self.zq + self.window_avg

        else:
            start_predict_idx = 0

        window_avg = self.window_avg
        window = self.window

        for i in range(start_predict_idx, X.size):
            value = X[i] - window_avg

            # Found an extreme value.
            if value > self.zq:
                extremes[i] = True
            else:
                # Update the window.
                self.n += 1
                thresholds[i] = self.zq + window_avg

                # Reestimate EVD's parameters and update the zq.
                if value > self.t:
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

                # Update the window average.
                window = np.append(window, X[i])
                window_avg += (window[-1] - window[0]) / self.depth
                window = window[1:]

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
    def _solve_grimshaw(lower_bound: float, upper_bound: float) -> np.ndarray:
        x0 = np.linspace(lower_bound, upper_bound, num_candidates, endpoint=True)
        results = minimize(
            _obj,
            x0,
            args=excesses,
            method="L-BFGS-B",
            bounds=[(lower_bound, upper_bound)] * num_candidates,
            jac=True,
        )

        return results.x

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
    candidates_1 = _solve_grimshaw(a + epsilon, -epsilon)
    candidates_2 = _solve_grimshaw(b, c)
    candidates = np.unique(np.concatenate([candidates_1, candidates_2]))

    # Calculate gamma and sigma.
    gamma = _v(excesses, candidates) - 1
    sigma = gamma / candidates

    # Include gamma and sigma for candidate = 0.
    gamma = np.append(gamma, 0.0)
    sigma = np.append(sigma, ymean)  # TODO: why ymean?

    # Calculate the log-likelihood and choose the best one.
    best_idx = np.nanargmax(_log_likelihood(excesses, gamma, sigma))

    return gamma[best_idx], sigma[best_idx]


@nb.njit([nb.bool(nb.float32), nb.bool(nb.float64)])
def _is0(a: float) -> bool:
    return abs(a) <= 1e-9


@nb.guvectorize(
    [
        (nb.float32[:], nb.float32, nb.float32, nb.float32[:]),
        (nb.float64[:], nb.float64, nb.float64, nb.float64[:]),
    ],
    "(n),(),()->()",
    nopython=True,
)
def _log_likelihood(excesses: np.ndarray, gamma: float, sigma: float, res: np.ndarray):
    "Log likelihood of the Generalized Pareto Distribution."
    Nt = excesses.size

    if _is0(gamma):
        res[0] = -Nt * np.log(sigma) - np.sum(excesses) / sigma
    else:
        x = gamma / sigma
        res[0] = -Nt * np.log(sigma) - (1.0 + 1.0 / gamma) * np.sum(
            np.log(1.0 + x * excesses)
        )


# @nb.njit(
#     [
#         nb.float32[:](nb.float32[:], nb.float32[:]),
#         nb.float64[:](nb.float64[:], nb.float64[:]),
#     ]
# )
def _v(excesses: np.ndarray, x: np.ndarray):
    one_plus_prod = 1.0 + x[..., None] * excesses[None, ...]
    return 1.0 + np.mean(np.log(one_plus_prod + 1e-9), axis=-1)


# @nb.njit(
#     [
#         nb.float32[:](nb.float32[:], nb.float32[:]),
#         nb.float64[:](nb.float64[:], nb.float64[:]),
#     ]
# )
def _obj(x: np.ndarray, excesses: np.ndarray):
    one_plus_prod = 1.0 + x[..., None] * excesses[None, ...]

    # Calculate the objective.
    u = np.mean(1.0 / one_plus_prod, axis=-1)
    v = _v(excesses, x)
    w = u * v - 1
    obj = np.sum(w**2)

    # Calculate the objective derivative wrt to each x.
    u_deriv = -np.mean(excesses[None, ...] / one_plus_prod**2, axis=-1)
    v_deriv = np.mean(excesses[None, ...] / one_plus_prod, axis=-1)
    obj_deriv = 2 * w * (u_deriv * v + u * v_deriv)

    return obj, obj_deriv


@nb.njit()
def _calculate_threshold(
    *, q: float, gamma: float, sigma: float, n: int, Nt: int, t: float
) -> float:
    tau = q * n / Nt

    if _is0(gamma):
        return t - sigma * log(tau)
    else:
        return t + (sigma / gamma) * (pow(tau, -gamma) - 1)
