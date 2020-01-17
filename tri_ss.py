"""
A module for representing triangular state space vector systems.

"""

class TriSS:
    """
    A class for representing the following triangular state space vector
    system:

        X_{1,t+1} = Θ_10 + Θ_11 * X_{1,t} + Λ_10 * W_{t+1}
        X_{2,t+1} = Θ_20 + Θ_21 * X_{1,t} + Θ_22 * X_{2,t}
                    + Θ_23 * (X_{1,t} ⊗ X_{1,t}) + Λ_20 W_{t+1}
                    + Λ_21 (X_{1,t} ⊗ W_{t+1}) + Λ_22 (W_{t+1} ⊗ W_{t+1})

    Parameters
    ----------
    Θ_10, Θ_11, Λ_10, Θ_20, Θ_21, Θ_22, Θ_23, Λ_20, Λ_21, Λ_22 : ndarray(float, ndim=2)
        See above.

    Attributes
    ----------
    Θ_10, Θ_11, Λ_10, Θ_20, Θ_21, Θ_22, Θ_23, Λ_20, Λ_21, Λ_22 : See above

    References
    ----------

    .. [1] Borovička, Jaroslav & Hansen, Lars Peter, 2014. "Examining
           macroeconomic models through the lens of asset pricing," Journal of
           Econometrics, Elsevier, vol. 183(1), pages 67-90.

    """

    def __init__(self, Θ_10, Θ_11, Λ_10, Θ_20, Θ_21, Θ_22, Θ_23, Λ_20, Λ_21,
                 Λ_22):
        self.Θ_10 = Θ_10
        self.Θ_11 = Θ_11
        self.Λ_10 = Λ_10
        self.Θ_20 = Θ_20
        self.Θ_21 = Θ_21
        self.Θ_22 = Θ_22
        self.Θ_23 = Θ_23
        self.Λ_20 = Λ_20
        self.Λ_21 = Λ_21
        self.Λ_22 = Λ_22


def map_perturbed_model_to_tri_ss(perturbed_model_params):
    """
    Maps parameters from the perburbed model into the triangular system.

    Parameters
    ----------
    perturbed_model_params : dict
        Dictionary containing the following keys corresponding to the
        perturbed model parameters: ψ_q, ψ_x, ψ_w, ψ_qq, ψ_xq, ψ_xx, ψ_wq,
        ψ_xw, ψ_ww

    Returns
    ----------
    tri_ss : TriSS
        The corresponding triangular state vector system represented as a
        `TriSS` object.

    References
    ----------

    .. [1] Borovička, Jaroslav & Hansen, Lars Peter, 2014. "Examining
           macroeconomic models through the lens of asset pricing," Journal of
           Econometrics, Elsevier, vol. 183(1), pages 67-90.

    """

    tri_ss_params = {
        'Θ_10': perturbed_model_params['ψ_q'],
        'Θ_11': perturbed_model_params['ψ_x'],
        'Λ_10': perturbed_model_params['ψ_w'],
        'Θ_20': perturbed_model_params['ψ_qq'],
        'Θ_21': 2 * perturbed_model_params['ψ_xq'],
        'Θ_22': perturbed_model_params['ψ_x'],
        'Θ_23': perturbed_model_params['ψ_xx'],
        'Λ_20': 2 * perturbed_model_params['ψ_wq'],
        'Λ_21': 2 * perturbed_model_params['ψ_xw'],
        'Λ_22': perturbed_model_params['ψ_ww']
        }

    tri_ss = TriSS(*tri_ss_params.values())

    return tri_ss
