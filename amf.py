"""
A module for representing additive and multiplicative functionals of
triangular state space vector systems.

"""

import numpy as onp
from util.utilities import *


# TODO(QBatista):
# - Avoid matrix inversions
# - Precompute kronecker products

class Amf:
    """
    A class for representing additive and multiplicative functionals whose
    additive increment follows:

        Y_{t+1} - Y_{t} = Γ_0 + Γ_1 * X_{1,t} + Γ_2 * X_{2,t}
                          + Γ_3 * (X_{1,t} ⊗ X_{1,t}) + Ψ_0 * W_{t+1}
                          + Ψ_1 * (X_{1,t} ⊗ W_{t+1})
                          + Ψ_2 * (W_{t+1} ⊗ W_{t+1})

    where X_{1,t}, X_{2,t} and W_{t+1} are elements of a triangular state
    vector system.

    Parameters
    ----------
    𝒫 : tuple
        A tuple containing the parameters of the additive increment in the
        following order: Γ_0, Γ_1, Γ_2, Γ_3, Ψ_0, Ψ_1, Ψ_2.

    tri_ss : TriSS
        An instance of a `TriSS` object representing the underlying triangular
        state vector system.

    α_h : ndarray(float, ndim=2), optional(default=None)
        An array representinng the shock exposure direction.

    Attributes
    ----------
    𝒫, tri_ss, α_h : See parameters.

    𝒫_t_bar_path, 𝒫_t_tilde_path : list
        Lists containing the values of 𝒫_bar and 𝒫_tilde respectively obtained
        from iterating on the following mappings:

            𝒫_{t}_tilde = 𝒫 + Ɛ_tilde(𝒫_{t-1}_bar)
            𝒫_{t}_bar = Ɛ_bar(𝒫_{t}_tilde)

    References
    ----------

    .. [1] Borovička, Jaroslav & Hansen, Lars Peter, 2014. "Examining
           macroeconomic models through the lens of asset pricing," Journal of
           Econometrics, Elsevier, vol. 183(1), pages 67-90.

    """


    def __init__(self, 𝒫, tri_ss):
        self.𝒫 = 𝒫
        self.tri_ss = tri_ss

        𝒫_0_bar = tuple(onp.zeros_like(x) for x in 𝒫)
        self.𝒫_t_bar_path = [𝒫_0_bar]
        self.𝒫_t_tilde_path = [None]
        self.Σ_t_tilde_path = [None]
        self.add_Σ_to_path = False

    def Ɛ_bar(self, 𝒫):
        """
        Ɛ_bar mapping (see appendix of reference [1]).

        Parameters
        ----------
        𝒫 : tuple
            A tuple containing the parameters of the additive increment in the
            following order: Γ_0, Γ_1, Γ_2, Γ_3, Ψ_0, Ψ_1, Ψ_2.

        Returns
        ----------
        𝒫_bar : tuple
            A tuple containing the following parameters: Γ_0_bar, Γ_1_bar,
            Γ_2_bar, Γ_3_bar, Ψ_0_bar, Ψ_1_bar, Ψ_2_bar.

        """

        # Unpack parameters
        Γ_0, Γ_1, Γ_2, Γ_3, Ψ_0, Ψ_1, Ψ_2 = 𝒫

        n, k = Γ_1.shape[1], Ψ_0.shape[1]

        # Compute 𝒫_bar
        Σ_inv = onp.eye(k) - sym(mat(2 * Ψ_2, (k, k)))
        Σ = onp.linalg.inv(Σ_inv)
        mat_Ψ_1 = mat(Ψ_1, (k, n))  # μ_1_t

        if self.add_Σ_to_path:
            self.Σ_t_tilde_path.append(Σ)

        Γ_0_bar = Γ_0 - 1 / 2 * onp.log(onp.linalg.det(Σ_inv)) + \
            1 / 2 * Ψ_0 @ Σ @ Ψ_0.T

        Γ_1_bar = Γ_1 + Ψ_0 @ Σ @ mat_Ψ_1

        Γ_3_bar = Γ_3 + 1 / 2 * vec(mat_Ψ_1.T @ Σ @ mat_Ψ_1).T

        𝒫_bar = (Γ_0_bar, Γ_1_bar, Γ_2, Γ_3_bar, onp.array([[0.]]),
                 onp.array([[0.]]), onp.array([[0.]]))

        return 𝒫_bar

    def Ɛ_tilde(self, 𝒫_bar):
        """
        Ɛ_tilde mapping (see appendix of reference [1]).

        Parameters
        ----------
        𝒫_bar : tuple
            A tuple containing the following parameters: Γ_0_bar, Γ_1_bar,
            Γ_2_bar, Γ_3_bar, Ψ_0_bar, Ψ_1_bar, Ψ_2_bar.

        Returns
        ----------
        𝒫_tilde : tuple
            A tuple containing the following parameters: Γ_0_tilde, Γ_1_tilde,
            Γ_2_tilde, Γ_3_tilde, Ψ_0_tilde, Ψ_1_tilde, Ψ_2_tilde.

        """

        # Unpack parameters
        Θ_10 = self.tri_ss.Θ_10
        Θ_11 = self.tri_ss.Θ_11
        Λ_10 = self.tri_ss.Λ_10
        Θ_20 = self.tri_ss.Θ_20
        Θ_21 = self.tri_ss.Θ_21
        Θ_22 = self.tri_ss.Θ_22
        Θ_23 = self.tri_ss.Θ_23
        Λ_20 = self.tri_ss.Λ_20
        Λ_21 = self.tri_ss.Λ_21
        Λ_22 = self.tri_ss.Λ_22

        n, k = Θ_10.shape[0], Λ_10.shape[0]

        Γ_0_bar, Γ_1_bar, Γ_2_bar, Γ_3_bar, Ψ_0_bar, Ψ_1_bar, Ψ_2_bar = 𝒫_bar

        # Compute 𝒫_tilde
        Γ_0_tilde = Γ_0_bar + Γ_1_bar @ Θ_10 + Γ_2_bar @ Θ_20 + \
            Γ_3_bar @ onp.kron(Θ_10, Θ_10)

        Γ_1_tilde = Γ_1_bar @ Θ_11 + Γ_2_bar @ Θ_21 + \
            Γ_3_bar @ (onp.kron(Θ_10, Θ_11) + onp.kron(Θ_11, Θ_10))

        Γ_2_tilde = Γ_2_bar @ Θ_22

        Γ_3_tilde = Γ_2_bar @ Θ_23 + Γ_3_bar @ onp.kron(Θ_11, Θ_11)

        Ψ_0_tilde = Γ_1_bar @ Λ_10 + Γ_2_bar @ Λ_20 + \
            Γ_3_bar @ (onp.kron(Θ_10, Λ_10) + onp.kron(Λ_10, Θ_10))

        # FIX HERE: Pre-compute
        temp = onp.hstack([onp.kron(Λ_10, Θ_11[:, [j]]) for j in range(n)])

        Ψ_1_tilde = Γ_2_bar @ Λ_21 + Γ_3_bar @ (onp.kron(Θ_11, Λ_10) + temp)

        Ψ_2_tilde = Γ_2_bar @ Λ_22 + Γ_3_bar @ onp.kron(Λ_10, Λ_10)

        𝒫_tilde = (Γ_0_tilde, Γ_1_tilde, Γ_2_tilde, Γ_3_tilde, Ψ_0_tilde,
                    Ψ_1_tilde, Ψ_2_tilde)

        return 𝒫_tilde

    def iterate(self, T):
        """
        Add T iterations on the following mapping to 𝒫_t_bar_path and
        𝒫_t_tilde_path:

            𝒫_{t}_tilde = 𝒫 + Ɛ_tilde(𝒫_{t-1}_bar)
            𝒫_{t}_bar = Ɛ_bar(𝒫_{t}_tilde)

        Parameters
        ----------
        T : scalar(int)
            Number of iterations.

        """

        self.add_Σ_to_path = True

        for _ in range(T):
            temp = zip(self.𝒫, self.Ɛ_tilde(self.𝒫_t_bar_path[-1]))

            𝒫_tilde = tuple(x + y for x, y in temp)
            𝒫_bar = self.Ɛ_bar(𝒫_tilde)

            self.𝒫_t_tilde_path.append(𝒫_tilde)
            self.𝒫_t_bar_path.append(𝒫_bar)

        self.add_Σ_to_path = False

    def 𝛆(self, x_1, t, α_h):
        """
        Compute shock elasticity for a given state tuple x and time period t.

        Parameters
        ----------
        x : tuple
            Tuple containing arrays of values for X_{1,t} and X_{2,t}.

        t : scalar(int)
            Time period.

        α_h : ndarray(float, ndim=2), optional(default=None)
            An array representinng the shock exposure direction.

        Returns
        ----------
        𝛆_x_t : scalar(float)
            Shock elasticity.

        """

        T = len(self.𝒫_t_tilde_path) - 1

        if t > T:
            self.iterate(t-T)

        Σ_t_tilde = self.Σ_t_tilde_path[t]
        _, Γ_1, _, _, Ψ_0, Ψ_1, _ = self.𝒫_t_tilde_path[t]
        n, k = Γ_1.shape[1], Ψ_0.shape[1]

        μ_0_t = Ψ_0

        μ_1_t = mat(Ψ_1, (k, n))

        𝛆_x_t = α_h.T @ Σ_t_tilde @ (μ_0_t.T + μ_1_t @ x_1.T)

        return onp.asscalar(𝛆_x_t)
