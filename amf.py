"""
A module for representing additive and multiplicative functionals of
triangular state space vector systems.

"""

import numpy as onp
from util.utilities import *
from numba import jit, njit


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
        x_1 : ndarray(float, dim=2)
            X_{1,t}.

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

    def eigenfunction(self, tol=1e-12, maxiters=5000):
        """
        Compute the eigenfunction for a given multiplicative functional
        iteration and triangularized set of state dynamics.

        Parameters
        ----------
        tol: scalar(float)
            The tolerance for numerically calculating the eigenfunction parameters

        maxiters: scalar(int)
            The maximum number of iterations for the above calculation.


        Returns
        ----------
        η: scalar(float)
            The principal eigenvalue
        Γ_1_bar: ndarray(float, dim=2)
            The eigenfunction coefficient on X^1_t
        Γ_2_bar: ndarray(float, dim=2)
            The eigenfunction coefficient on X^2_t
        Γ_3_bar ndarray(float, dim=2)
            The eigenfunction coefficient on (X^1_t ⊗ X^1_t)

        """

        Γ_0, Γ_1, Γ_2, Γ_3, Ψ_0, Ψ_1, Ψ_2 = self.𝒫

        Γ_2_bar = onp.linalg.solve((onp.eye(len(self.tri_ss.Θ_22)) - self.tri_ss.Θ_22).T, Γ_2.T).T

        Θ_10 = self.tri_ss.Θ_10
        Θ_11 = self.tri_ss.Θ_11
        Θ_23 = self.tri_ss.Θ_23
        Λ_10 = self.tri_ss.Λ_10
        Λ_21 = self.tri_ss.Λ_21
        Λ_22 = self.tri_ss.Λ_22

        n, k = Θ_10.shape[0], Λ_10.shape[1]
        temp = onp.hstack([onp.kron(Λ_10, Θ_11[:, [j]]) for j in range(n)])

        @jit
        def iteration(Γ_3_bar):

            Ψ_1_tilde = Γ_2_bar @ Λ_21 + Γ_3_bar @ (onp.kron(Θ_11, Λ_10) + temp)
            Ψ_2_tilde = Γ_2_bar @ Λ_22 + Γ_3_bar @ onp.kron(Λ_10, Λ_10)
            Γ_3_tilde = Γ_2_bar @ Θ_23 + Γ_3_bar @ onp.kron(Θ_11, Θ_11)

            Σ_inv = onp.eye(k) - sym(mat(2 * (Ψ_2_tilde + Ψ_2), (k, k)))
            Σ = onp.linalg.inv(Σ_inv)

            mat_Ψ_1 = mat(Ψ_1_tilde + Ψ_1, (k, n))
            Γ_3_bar = Γ_3 + Γ_3_tilde + 1 / 2 * vec(mat_Ψ_1.T @ Σ @ mat_Ψ_1).T

            return Γ_3_bar

        Γ_3_bar = onp.ones_like(Γ_3)
        diff = 1
        iters = 0
        while diff > tol and iters < maxiters:
            iters += 1
            Γ_3_bar_new = iteration(Γ_3_bar)
            # diff = onp.abs(onp.max(onp.max((Γ_3_bar - Γ_3_bar_new)/Γ_3_bar)))
            diff = onp.abs(onp.max(onp.max(Γ_3_bar - Γ_3_bar_new)))
            Γ_3_bar = onp.copy(Γ_3_bar_new)
        if diff > tol:
            raise ValueError(f"Did not converge after {maxiters} iterations. Difference is {diff}.")
        else:
            # print(f"Converged after {iters} iterations.")
            pass

        n, k = Θ_10.shape[0], Λ_10.shape[1]
        temp = onp.hstack([onp.kron(Λ_10, Θ_11[:, [j]]) for j in range(n)])
        Ψ_1_tilde = Γ_2_bar @ Λ_21 + Γ_3_bar @ (onp.kron(Θ_11, Λ_10) + temp)
        Ψ_2_tilde = Γ_2_bar @ Λ_22 + Γ_3_bar @ onp.kron(Λ_10, Λ_10)
        C1 = Γ_2_bar @ self.tri_ss.Θ_21 + Γ_3_bar @ (onp.kron(Θ_10, Θ_11) + onp.kron(Θ_11, Θ_10))
        Σ_inv = onp.eye(k) - sym(mat(2 * (Ψ_2_tilde + Ψ_2), (k, k)))
        Σ = onp.linalg.inv(Σ_inv)
        A = Σ @ mat(Ψ_1_tilde + Ψ_1, (k, n))
        C2 = Γ_2_bar @ self.tri_ss.Λ_20 + Γ_3_bar @ (onp.kron(Θ_10, Λ_10) + onp.kron(Λ_10, Θ_10))

        Γ_1_bar = onp.linalg.solve((onp.eye(len(Θ_11)) - Θ_11 - Λ_10 @ A).T, (Γ_1 + C1 + Ψ_0@A + C2@A).T).T

        Ψ_0_tilde = Γ_1_bar @ Λ_10 + C2

        C1 = - 1 / 2 * onp.log(onp.linalg.det(Σ_inv)) + \
                    1 / 2 * (Ψ_0 + Ψ_0_tilde) @ Σ @ (Ψ_0 + Ψ_0_tilde).T
        C2 = Γ_1_bar @ Θ_10 + Γ_2_bar @ self.tri_ss.Θ_20 + Γ_3_bar @ onp.kron(Θ_10, Θ_10)
        η = Γ_0 + C1 + C2

        return η, Γ_1_bar, Γ_2_bar, Γ_3_bar

    def infinite_sum(self, X_1_t, X_2_t, N=500):
        """
        Compute the infinite sum of expectations for Y_{t+\tau} - Y_t

        Parameters
        ----------
        X_1_t: ndarray(float, dim=2)
            Values for X_1

        X_2_t: ndarray(float, dim=2)
            Values for X_2

        N: scalar(int)
            The number of periods to sum before using the eigenfunction approximation

        Returns
        ----------
        π: scalar(float)
            The infinite sum of expectations. For example, the price-dividend
            ratio under the correct specification of Y_{t+1}-Y_t

        """
        # Ensure that the switching point between calculated coefficients and limiting coefficients is okay
        if N > len(self.P_t_bar_path):
            self.iterate(N - len(self.P_t_bar_path))
        π = 0
        # Iterate over the N first calculated terms
        for i in range(N):
            P_i = self.P_t_bar_path[i]
            π_current = onp.exp(P_i[0] + P_i[1] @ X_1_t + P_i[2] @ X_2_t + P_i[3] @ onp.kron(X_1_t, X_1_t))
            π += π_current
        η, Γ_1_bar, Γ_2_bar, Γ_3_bar = self.eigenfunction()
        π += onp.exp(P_i[0] + η + Γ_1_bar @ X_1_t + \
                      Γ_2_bar @ X_2_t + Γ_3_bar @ onp.kron(X_1_t, X_1_t)) / (1 - onp.exp(η))
        return π

    def infinite_sum_derivative(self, x_1, x_2, α_h, N=10000):
        "Test function only - ignore"
        if N > len(self.P_t_bar_path):
            self.iterate(N - len(self.P_t_bar_path))
        numerator = 0
        for i in range(N):
            P_i = self.P_t_bar_path[i]
            numerator += onp.exp(P_i[0] + P_i[1] @ x_1 + P_i[2] @ x_2 + \
                                P_i[3] @ onp.kron(x_1, x_1)) * self.𝛆(onp.array([x_1]), i+1, α_h)
        return numerator/self.infinite_sum(x_1, x_2)

    def limiting_elasticity():
        raise NotImplementedError()

    def example_2_elasticity(self, ρ, q, η_0_vmc, η_1_vmc, η_2_vmc, A,\
                             B, C, X_1_t):

        """
        This function computes

        d/dr log E [(C_{t+1}S_{t+1})/(C_tS_t) * A_{t+1}/C_{t+1} * H_{t+1}(r)]|r=0

        In order for the function to work 𝒫 must have been specified to contain
        the dynamics for (C_{t+1}S_{t+1})/(C_tS_t).

        This function assumes v_t - c_t ≈ η_0_vmc + q(η_1_vmc + A X_1_t) +
                          q^2/2 (η_2_vmc + A X_2_t + B X_1_t + C (X_1_t⊗X_1_t))
        and that a_t - c_t = -log(1-𝛽) + (1-ρ)*(v_t - c_t).

        Parameters
        ----------
        ρ: float
            The inverse EIS for the system

        q: float
            The perturbation parameter

        η_0_vmc, η_1_vmc, η_2_vmc, A, B, C
            The parameters of the approximation for v_t - c_t. See above in
            docstring.

        X_1_t: ndarray(float)
            The first order approximation for the state vector

        Returns
        ----------
        elas: float
            The derivative specified above in the docstring.
        """

        Γ_0, Γ_1, Γ_2, Γ_3, Ψ_0, Ψ_1, Ψ_2 = self.𝒫
        n, k = Γ_1.shape[1], Ψ_0.shape[1]
        Θ_10 = self.tri_ss.Θ_10 #psi_q
        Θ_11 = self.tri_ss.Θ_11 #psi_x
        Λ_10 = self.tri_ss.Λ_10 #psi_w
        Λ_20 = self.tri_ss.Λ_20 #2psi_wq
        Λ_21 = self.tri_ss.Λ_21 #2psi_xw
        Λ_22 = self.tri_ss.Λ_22 #psi_ww

        A_a_1 = (1 - ρ) * (q * A @ Λ_10 + \
                           q**2/2 * (A@Λ_20 + B@Λ_10 + \
                                     C@(onp.kron(Θ_10, Λ_10) + \
                                                    onp.kron(Λ_10, Θ_10))))
        A_a_2 = (1 - ρ) * q**2/2 * (mat(A@Λ_21, (k,n)).T + \
                                    mat(C@onp.kron(Θ_11,Λ_10), (k,n)).T +\
                                    mat(C@onp.kron(Λ_10,Θ_11), (n,k)))
        A_a_3 = (1 - ρ) * q**2/2 * (A@Λ_22 + C@onp.kron(Λ_10,Λ_10))

        A = Ψ_0 + A_a_1 + X_1_t@(mat(Ψ_1, (k,n)).T + A_a_2)
        B = Ψ_2 + A_a_3

        invert_term = onp.eye(k) - sym(mat(2*B, (k,k)))

        elas = onp.linalg.solve(invert_term, A.T)

        return elas
