import numpy as np

class UnadjustedLangevin:
    """
    Unadjusted Langevin Algorithm (ULA) for sampling from a target
    density f(x) \propto exp(-V(x)) in R^d.

    Inputs:
    gradV:
        Gradient of the potential V(x). Should accept an array x of shape (d,)
        and return an array of the same shape.
    step_size: float
        Langevin step size > 0.
    pi0_sampler:
        Function that returns a single draw X^(0) from the initial distribution π_0.
    h: optional
        Test function to apply to samples when using `estimate`.
        Should accept an array of shape (d,) or (N, d) and return values accordingly.
    """

    def __init__(self, gradV, step_size, pi0_sampler, h=None):
        self.gradV = gradV
        self.eps = step_size
        self.pi0_sampler = pi0_sampler
        self.h = h

    def sample_chain(self, N, x0=None):
        """
        Generate a length-N Markov chain {X^(n)} using ULA.

        Parameters:
        N: int
            Number of samples to generate.
        x0: array-like, optional
            Initial state X^(0). If None, a draw from pi0_sampler is used.

        Output:
        X: ndarray, shape (N, d)
            The generated chain.
        """

        if x0 is None:
            x = np.asarray(self.pi0_sampler())
        else:
            x = np.asarray(x0)

        x = np.atleast_1d(x)
        d = x.shape[0]
        X = np.empty((N, d), dtype=float)

        for n in range(N):
            X[n] = x
            noise = np.random.normal(size=d)
            x = x - self.eps * self.gradV(x) + np.sqrt(2.0 * self.eps) * noise

        return X

    def estimate(self, N, burn_in=0, thinning=1, x0=None):
        """
        Monte Carlo estimate of E_f[h(X)] using ULA samples.

        Parameters:
        N: int
            Number of post-burn-in samples to use in the average.
        burn_in: int, optional
            Number of initial iterations to discard.
        thinning: int, optional
            Keep every `thinning`-th sample after burn-in.
        x0: array-like, optional
            Initial state X^(0).

        Outputs: Estimate of E_f[h(X)].
        """

        if self.h is None:
            raise ValueError("Test function h was not provided at initialization.")

        # total chain length needed
        total_steps = burn_in + N * thinning
        chain = self.sample_chain(total_steps, x0=x0)

        # discard burn-in and thin
        chain_post = chain[burn_in::thinning][:N]

        values = self.h(chain_post)
        return np.mean(values)
