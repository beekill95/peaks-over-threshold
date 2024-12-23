from __future__ import annotations

from math import log

import numpy as np
import numba as nb
from scipy.optimize import minimize
from tqdm.auto import tqdm


class SPOT:
    def __init__(self, q: float = 1e-3) -> None:
        self.q = q
        self.calibrated = False
        self.t = 0
        self.zq = 0

        self.n = 0
        self.excesses = np.zeros(1, dtype=float)
        self._grimshaw = None

    def fit(self, X: np.ndarray, init_quantile: float = 0.98):
        """
        Calibrate the algorithm.
        """
        self._grimshaw = _Grimshaw()
        zq, t = pot(X, self.q, init_quantile=init_quantile, _grimshaw=self._grimshaw)
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
        for i in tqdm(range(start_predict_idx, X.size)):
            value = X[i]

            # Found an extreme value.
            if value > self.zq:
                extremes[i] = True
            else:
                self.n += 1

                if value > self.t:
                    # Reestimate EVD's parameters and update the zq.
                    self.excesses = np.append(self.excesses, value - self.t)

                    gamma, sigma = self._grimshaw.solve(self.excesses)
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
        self._grimshaw = None

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
        self._grimshaw = _Grimshaw()
        zq, t = pot(diff, self.q, init_quantile=init_quantile, _grimshaw=self._grimshaw)
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

        for i in tqdm(range(start_predict_idx, X.size)):
            value = X[i] - window_avg
            thresholds[i] = self.zq + window_avg

            # Found an extreme value.
            if value > self.zq:
                extremes[i] = True
            else:
                # Update the window.
                self.n += 1

                # Reestimate EVD's parameters and update the zq.
                if value > self.t:
                    self.excesses = np.append(self.excesses, value - self.t)

                    gamma, sigma = self._grimshaw.solve(self.excesses)
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
    _grimshaw: _Grimshaw | None = None,
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

    if _grimshaw is None:
        _grimshaw = _Grimshaw()

    gamma, sigma = _grimshaw.solve(excesses)

    zq = _calculate_threshold(
        q=q, gamma=gamma, sigma=sigma, n=X.size, Nt=excesses.size, t=t
    )

    return zq, t


class _Grimshaw:
    def __init__(self, num_candidates: int = 10, epsilon: float = 1e-8) -> None:
        self.num_candidates = num_candidates
        self.epsilon = epsilon

        self.candidates_1 = None
        self.candidates_2 = None

    def solve(self, excesses: np.ndarray) -> tuple[float, float]:
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
        epsilon = self.epsilon
        candidates_1 = self._find_roots(
            excesses=excesses,
            lower_bound=a + epsilon,
            upper_bound=-epsilon,
            prev_candidates=self.candidates_1,
        )
        candidates_2 = self._find_roots(
            excesses=excesses,
            lower_bound=b,
            upper_bound=c,
            prev_candidates=self.candidates_2,
        )
        candidates = np.unique(np.concatenate([candidates_1, candidates_2]))

        # Calculate gamma and sigma.
        gamma = _v(excesses, candidates) - 1
        sigma = gamma / candidates

        # Include gamma and sigma for candidate = 0.
        gamma = np.append(gamma, 0.0)
        sigma = np.append(sigma, ymean)  # TODO: why ymean?

        # Calculate the log-likelihood and choose the best one.
        best_idx = np.nanargmax(_log_likelihood(excesses, gamma, sigma))

        # Store the candidates for later.
        self.candidates_1 = candidates_1
        self.candidates_2 = candidates_2

        return gamma[best_idx], sigma[best_idx]

    def _find_roots(
        self,
        *,
        excesses: np.ndarray,
        lower_bound: float,
        upper_bound: float,
        prev_candidates: np.ndarray | None,
    ) -> np.ndarray:
        num_candidates = self.num_candidates

        x0 = np.linspace(lower_bound, upper_bound, num_candidates, endpoint=True)
        if prev_candidates is not None:
            x0 = np.where(
                (lower_bound <= prev_candidates) & (prev_candidates <= upper_bound),
                prev_candidates,
                x0,
            )

        results = minimize(
            _obj2,
            x0,
            args=excesses,
            method="L-BFGS-B",
            bounds=[(lower_bound, upper_bound)] * num_candidates,
            jac=True,
        )

        return results.x


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


def _v(excesses: np.ndarray, x: np.ndarray):
    one_plus_prod = 1.0 + x[..., None] * excesses[None, ...]
    return 1.0 + np.nanmean(np.log(one_plus_prod), axis=-1)


@nb.njit(nogil=True)
def _calc_one_plus(x: np.ndarray, excesses: np.ndarray):
    return 1.0 + x[..., None] * excesses[None, ...]


@nb.njit(parallel=True, nogil=True, cache=True)
def _obj2(x: np.ndarray, excesses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    one_plus_prod = _calc_one_plus(x, excesses)

    Nt = excesses.size

    u = np.zeros_like(x)
    v = np.ones_like(x)
    u_deriv = np.zeros_like(x)
    v_deriv = np.zeros_like(x)

    for i in nb.prange(Nt):
        opp = one_plus_prod[:, i]
        u += 1.0 / (opp * Nt)
        v += np.log(opp) / Nt

        temp = excesses[i] / (Nt * opp)
        v_deriv += temp
        u_deriv -= temp / opp

    w = u * v - 1

    return np.sum(w**2), 2 * w * (u_deriv * v + u * v_deriv)


@nb.njit()
def _calculate_threshold(
    *, q: float, gamma: float, sigma: float, n: int, Nt: int, t: float
) -> float:
    tau = q * n / Nt

    if _is0(gamma):
        return t - sigma * log(tau)
    else:
        return t + (sigma / gamma) * (pow(tau, -gamma) - 1)
