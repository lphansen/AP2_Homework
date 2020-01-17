import pandas as pd
import autograd.numpy as np
from autograd import jacobian as jacrev
import numpy as onp
from IPython.display import display, Math, Latex
from numba import jit
import matplotlib.pyplot as plt
from scipy.io import loadmat

def mat(vec, shape):
    M = vec.reshape(shape, order='F')
    return M

def vec(M):
    v = M.reshape((-1, 1), order='F')
    return v

def sym(M):
    sym_M = (M + M.T) / 2
    return sym_M

μ = 0.0015
μ_d = 0.0015
α = 0.979
σ = 0.0078
ϕ = 3.0
ϕ_e = 0.044
ϕ_d = 4.5
ν_1 = 0.987
σ_w = 0.23 * 1e-5
# σ_w = -0.038 * σ
n = 2  # Number of state variables
m = 4  # Number of shock variables

σ_squared = σ ** 2

@jit
def ψ(X, W, h):
    # Unpack
    x_t, σ_t_squared = X.ravel()
    e_tp1, w_tp1, η_tp1, u_tp1 = W.ravel()
    h = h.ravel()

    x_next = α * x_t + ϕ_e * np.sqrt(σ_t_squared) * e_tp1
    σ_squared_next = σ_squared + ν_1 * (σ_t_squared - σ_squared) + σ_w * w_tp1

    X_next = np.array([[x_next], [σ_squared_next]])

    return X_next

@jit
def compute_derivatives(f):
    # First-order
    df_x = jacrev(f, argnum=0)
    df_w = jacrev(f, argnum=1)
    df_h = jacrev(f, argnum=2)

    # Second-order
    df_xx = jacrev(df_x, argnum=0)
    df_xw = jacrev(df_x, argnum=1)
    df_xh = jacrev(df_x, argnum=2)

    df_ww = jacrev(df_w, argnum=1)
    df_wh = jacrev(df_w, argnum=2)

    df_hh = jacrev(df_h, argnum=2)

    return df_x, df_w, df_h, df_xx, df_xw, df_xh, df_ww, df_wh, df_hh

@jit
def g_c(X, W, h):
    # Unpack
    x_t, σ_t_squared = X.ravel()
    e_tp1, w_tp1, η_tp1, u_tp1 = W.ravel()

    g_c_next = μ + x_t + np.sqrt(σ_t_squared) * η_tp1

    return g_c_next

@jit
def g_g(X, W, h):
    # Unpack
    x_t, σ_t_squared = X.ravel()
    e_tp1, w_tp1, η_tp1, u_tp1 = W.ravel()

    g_g_next = μ_d + ϕ * x_t + ϕ_d * np.sqrt(σ_t_squared) * u_tp1

    return g_g_next

def derivs_BY(growth_variable, ρ = 1.0, γ_o = 1.5, δ = .998):

    dψ_x, dψ_w, dψ_h, dψ_xx, dψ_xw, dψ_xh, dψ_ww, dψ_wh, dψ_hh = compute_derivatives(ψ)
    dg_c_x, dg_c_w, dg_c_h, dg_c_xx, dg_c_xw, dg_c_xh, dg_c_ww, dg_c_wh, dg_c_hh = compute_derivatives(g_c)
    dg_g_x, dg_g_w, dg_g_h, dg_g_xx, dg_g_xw, dg_g_xh, dg_g_ww, dg_g_wh, dg_g_hh = compute_derivatives(g_g)

    # Initial conditions
    X_0 = np.array([0, σ**2])
    W_0 = np.zeros((m, 1))
    h_0 = np.zeros(1)

    # ψ derivatives
    ψ_x = dψ_x(X_0, W_0, h_0).reshape((n, -1))
    ψ_w = dψ_w(X_0, W_0, h_0).reshape((n, -1))
    ψ_h = dψ_h(X_0, W_0, h_0).reshape((n, -1))


    ψ_xx = dψ_xx(X_0, W_0, h_0).reshape((n, -1))
    ψ_xw = dψ_xw(X_0, W_0, h_0).reshape((n, -1))
    ψ_xh = dψ_xh(X_0, W_0, h_0).reshape((n, -1))

    ψ_ww = dψ_ww(X_0, W_0, h_0).reshape((n, -1))
    ψ_wh = dψ_wh(X_0, W_0, h_0).reshape((n, -1))

    ψ_hh = dψ_hh(X_0, W_0, h_0).reshape((n, -1))

    # g_c derivatives
    g_c_x = dg_c_x(X_0, W_0, h_0).reshape((1, -1))
    g_c_w = dg_c_w(X_0, W_0, h_0).reshape((1, -1))
    g_c_h = dg_g_h(X_0, W_0, h_0)

    g_c_xx = dg_c_xx(X_0, W_0, h_0).reshape((1, -1))
    g_c_xw = dg_c_xw(X_0, W_0, h_0).reshape((1, -1))
    g_c_xh = dg_c_xh(X_0, W_0, h_0).reshape((1, -1))

    g_c_ww = dg_c_ww(X_0, W_0, h_0).reshape((1, -1))
    g_c_wh = dg_c_wh(X_0, W_0, h_0).reshape((1, -1))

    g_c_hh = dg_c_hh(X_0, W_0, h_0)

    # g_g derivatives
    g_g_x = dg_g_x(X_0, W_0, h_0).reshape((1, -1))
    g_g_w = dg_g_w(X_0, W_0, h_0).reshape((1, -1))
    g_g_h = dg_c_h(X_0, W_0, h_0)

    g_g_xx = dg_g_xx(X_0, W_0, h_0).reshape((1, -1))
    g_g_xw = dg_g_xw(X_0, W_0, h_0).reshape((1, -1))
    g_g_xh = dg_g_xh(X_0, W_0, h_0).reshape((1, -1))

    g_g_ww = dg_g_ww(X_0, W_0, h_0).reshape((1, -1))
    g_g_wh = dg_g_wh(X_0, W_0, h_0).reshape((1, -1))

    g_g_hh = dg_g_hh(X_0, W_0, h_0).reshape((1, -1))

    η_0_c = g_c(X_0, W_0, h_0)
    η_0_vmc = 1 / (1 - ρ) * np.log((1 - δ) / (1 - δ * np.exp((1 - ρ) * η_0_c)))

    D_1_c = g_c_x.T
    F_1_c = g_c_w.T

    λ = δ * np.exp((1 - ρ) * η_0_c)

    a = np.eye(2) - λ * ψ_x.T
    b = λ * D_1_c
    S_1_vmc = np.linalg.solve(a, b)

    σ_1_v = F_1_c + ψ_w.T @ S_1_vmc
    η_1_c = g_c_h
    η_1_vmc = λ / (1 - λ) * ((1 - γ_o) * np.linalg.norm(σ_1_v, 2) ** 2 / 2 + η_1_c)# + S_1_vmc.T @ ψ_h)

    E_w = (1 - γ_o) * σ_1_v
    E_ww = (np.eye(m) + (1 + γ_o) ** 2 * (S_1_vmc.T @ ψ_w).T @ S_1_vmc.T @ ψ_w).reshape((-1, 1))

    A = S_1_vmc.T
    a = (np.eye(n ** 2) - λ * np.kron(ψ_x, ψ_x)).T
    b = (λ * A @ ψ_xx + λ* g_c_xx + (1 - ρ) * (1 - λ) / λ * np.kron(S_1_vmc, S_1_vmc).T).T
    B = np.linalg.solve(a, b).T

    first_term_C = λ * E_w.T @ mat(2 * A @ ψ_xw, (m, n)) + 2 * λ * A @ ψ_xh
    second_term_C = λ * B @ (np.kron(ψ_x, ψ_h) + np.kron(ψ_h, ψ_x))
    third_term_C = λ * E_w.T @ (
                                mat(B @ np.kron(ψ_x, ψ_w), (m, n)) +
                                mat(B @ np.kron(ψ_w, ψ_x), (n, m)).T
                                )
    fourth_term_C = λ * 2 * g_c_xh + λ * E_w.T @ mat(2 * g_c_xw, (m, n)) + \
        (1 - ρ) * (1 - λ) / λ * 2 * η_1_vmc * S_1_vmc.T

    b = (first_term_C + second_term_C + third_term_C + fourth_term_C).T
    a = (np.eye(n) - λ * ψ_x).T

    C = np.linalg.solve(a, b).T

    # first_term_D = A @ (ψ_ww @ E_ww + 2 * ψ_wh @ E_w + ψ_hh)
    # second_term_D = B @ (np.kron(ψ_h, ψ_h)
    #     + np.kron(ψ_w, ψ_w) @ E_ww
    #     + (np.kron(ψ_w, ψ_h) + np.kron(ψ_h, ψ_w)) @ E_w
    #     )
    # third_term_D = C @ (ψ_w @ E_w + ψ_h)
    # fourth_term_D = g_c_ww @ E_ww + 2 * g_c_wh @ E_w + g_c_hh
    # fifth_term_D = (1 - ρ) / λ * η_1_vmc ** 2
    #
    # D = λ / (1 - λ) * (first_term_D + second_term_D + third_term_D + fourth_term_D) + \
    #     fifth_term_D
    #
    η_0_g = g_g(X_0, W_0, h_0)
    # η_0_s = onp.log(δ) - ρ * η_0_c
    # η_0_q = η_0_s + η_0_g
    #
    # π_0 = η_0_q - np.log(1 - np.exp(η_0_q))
    #
    # # Match x term
    # a = np.eye(n) - np.exp(η_0_q)*ψ_x.T
    # b = (g_g_x - ρ*g_c_x).T
    # π_x = np.linalg.solve(a, b)
    #
    # # Match c term
    # a_1 = S_1_vmc.T@ψ_w + g_c_w
    # b_1 = -(1-γ_o)/2*np.linalg.norm(a_1,2)**2
    # a = 1-np.exp(η_0_q)
    # b = (ρ-1)*(a_1@E_w+b_1) + (g_g_w-ρ*g_c_w)@E_w + (g_g_h-ρ*g_c_h)\
    # + np.exp(η_0_q)*π_x.T@(ψ_w@E_w+ψ_h)
    # π_h = b/a
    #
    # D_1 = np.kron(a_1,a_1)@E_ww + 2*b_1*a_1@E_w + b_1**2

    # Linear term 2
    # Coefficient of x^2, xx, x, c: A_2,B_2,C_2,D_2, respectively
    a_2 = 2*A@ψ_wh + B@(np.kron(ψ_w,ψ_h)+np.kron(ψ_h,ψ_w)) + C@ψ_w + 2*g_c_wh
    b_2 = 2*mat(A@ψ_xw,(m,n)) + mat(B@np.kron(ψ_x,ψ_w),(m,n)) + mat(B@np.kron(ψ_w,ψ_x),(n,m)).T\
    + 2*mat(g_c_xw,(m,n))
    c_2 = A@ψ_ww + B@np.kron(ψ_w,ψ_w) + g_c_ww
    d_2 = -a_2@E_w - c_2@E_ww
    #
    # a_3 = a_1@(np.eye(m)+E_w@E_w.T)@b_2 + b_1*E_w.T@b_2 - (a_1@E_w+b_1)*E_w.T@b_2
    # b_3 = (vec(a_1.T@a_2).T+b_1*c_2)@E_ww + (d_2@a_1 + b_1*a_2)@E_w + b_1*d_2
    #
    # C_2 = a_3
    # D_2 = b_3
    #
    # # Linear term 3
    # # Coefficient of x^2, xx, x, c: A_3,B_3,C_3,D_3, respectively
    # χ = (1-ρ)/(1-γ_o)
    # a_4 = g_g_x - ρ*g_c_x + np.exp(η_0_q)*π_x.T@ψ_x
    # b_4 = g_g_w - ρ*g_c_w + np.exp(η_0_q)*π_x.T@ψ_w
    # c_4 = g_g_h - ρ*g_c_h + np.exp(η_0_q)*(π_x.T@ψ_h + π_h)
    #
    # a_5 = (0.5*a_2-χ*a_1)@E_w@a_4 + 0.5*b_4@b_2 + 0.5*c_2@E_ww@a_4 + (0.5*d_2-χ*b_1)*a_4
    # b_5 = vec((0.5*a_2-χ*a_1).T@b_4).T@E_ww + c_4*(0.5*a_2-χ*a_1)@E_w + 0.5*c_4*c_2@E_ww\
    # + (0.5*d_2-χ*b_1)*(b_4@E_w+c_4)
    #
    # C_3 = a_5
    # D_3 = b_5
    #
    # # Linear term 4
    # # Coefficient of x^2, xx, x, c: A_4,B_4,C_4,D_4, respectively
    #
    # a_6 = g_g_x - ρ*g_c_x + np.exp(π_0)/(1+np.exp(π_0))*π_x.T@ψ_x
    #
    # # π_xx
    #
    # # b_6 = g_g_xx - ρ*g_c_xx + np.exp(π_0)/(1+np.exp(π_0))**2 * vec(2*ψ_x.T@π_x@π_x.T@ψ_x).T\
    # # + np.exp(π_0)/(1+np.exp(π_0))*(π_x.T@ψ_xx + π_xx@np.kron(ψ_x,ψ_x))
    #
    # b_6_1 = g_g_xx - ρ*g_c_xx + np.exp(π_0)/(1+np.exp(π_0))**2 * vec(ψ_x.T@π_x@π_x.T@ψ_x).T\
    # + np.exp(π_0)/(1+np.exp(π_0))*π_x.T@ψ_xx
    #
    # b_6_2 = np.exp(π_0)/(1+np.exp(π_0)) * np.kron(ψ_x,ψ_x)
    #
    # # π_xh,π_xx,π_xh
    #
    # c_6_1 = 2*E_w.T@mat(g_g_xw-ρ*g_c_xw,(m,n)) + 2*g_g_xh - 2*ρ*g_c_xh\
    # + np.exp(π_0)/(1+np.exp(π_0))**2 * (2*E_w.T@ψ_w.T@π_x@π_x.T@ψ_x + 2*(π_x.T@ψ_h+π_h)@π_x.T@ψ_x)\
    # + np.exp(π_0)/(1+np.exp(π_0))*(E_w.T@mat(2*π_x.T@ψ_xw,(m,n)) + 2*π_x.T@ψ_xh)
    #
    # # c_6_2 calculated below
    #
    # c_6_3 = np.exp(π_0)/(1+np.exp(π_0)) * ψ_x
    #
    # # π_xx,π_xh,π_hh
    #
    # d_6_1 = (g_g_ww - ρ*g_c_ww)@E_ww + 2*(g_g_wh-ρ*g_c_wh)@E_w + g_g_hh-ρ*g_c_hh\
    # + np.exp(π_0)/(1+np.exp(π_0))**2 * (2*(π_x.T@ψ_h+π_h)@π_x.T@ψ_w@E_w+(π_x.T@ψ_h+π_h)**2+\
    #                                     vec(ψ_w.T@π_x@π_x.T@ψ_w).T@E_ww)\
    # + np.exp(π_0)/(1+np.exp(π_0)) * (π_x.T@ψ_ww@E_ww+2*π_x.T@ψ_wh@E_w+π_x.T@ψ_hh)
    #
    # d_6_2 = np.exp(π_0)/(1+np.exp(π_0)) * (np.kron(ψ_h,ψ_h) + np.kron(ψ_w,ψ_w)@E_ww + (np.kron(ψ_w,ψ_h)+np.kron(ψ_h,ψ_w))@E_w)
    #
    # d_6_3 = np.exp(π_0)/(1+np.exp(π_0)) * (ψ_w@E_w+ψ_h)
    #
    # d_6_4 = np.exp(π_0)/(1+np.exp(π_0))
    #
    # # Linear term 4
    # a_7 = vec(a_4.T@a_4).T
    # b_7 = 2 * c_4 * a_4 + 2 * E_w.T @ b_4.T @ a_4
    # c_7 = 2 * c_4 * b_4 @ E_w + vec(b_4.T@b_4).T @ E_ww + c_4**2
    #
    # a_8 = vec(π_x@π_x.T).T
    # b_8 = 2*π_h@π_x.T
    # c_8 = π_h**2
    #
    # a = np.eye(np.kron(ψ_x,ψ_x).shape[0]) - np.exp(π_0)/(1+np.exp(π_0)) * np.kron(ψ_x,ψ_x)
    # b = g_g_xx - ρ*g_c_xx + np.exp(π_0)/(1+np.exp(π_0))**2  * vec(ψ_x.T@π_x@π_x.T@ψ_x).T\
    # + np.exp(π_0)/(1+np.exp(π_0))*π_x.T@ψ_xx + a_7 - a_8
    # π_xx = np.linalg.solve(a.T, b.T).T
    #
    # c_6_2 = np.exp(π_0)/(1+np.exp(π_0)) * (π_xx @ (np.kron(ψ_x,ψ_h)+np.kron(ψ_h,ψ_x))+\
    #                                        E_w.T@(mat(π_xx @ np.kron(ψ_x,ψ_w),(m,n)) + \
    #                                               mat(π_xx @ np.kron(ψ_w,ψ_x),(n,m)).T))
    #
    # a = np.eye(ψ_x.shape[0]) - c_6_3
    # b = -χ*(1 - γ_o)**2 * a_3 + 2 * (1 - γ_o) * a_5 + c_6_1 + c_6_2 + b_7 - b_8
    # π_xh = np.linalg.solve(a.T, b.T).T
    #
    # a = 1 - d_6_4
    # b = χ**2*(1 - γ_o)**2*D_1 - χ*(1 - γ_o)**2*b_3 + 2*(1-γ_o)*b_5 + d_6_1 + π_xx@d_6_2 + π_xh@d_6_3 + c_7 - c_8
    # π_hh = b/a

    h = 1
    k1 = h * (ρ - 1) + 1 - γ_o
    k2 = h**2 / 2 * (ρ - 1) + h * (1-γ_o) / 2

    if growth_variable == "C":
        Γ_0_G = onp.array(η_0_c + g_c_h)
        Γ_1_G = onp.array(g_c_x + g_c_xh)
        Γ_2_G = onp.array(h**2 / 2 * g_c_x)
        Γ_3_G = onp.array(g_c_xx)
        Ψ_0_G = onp.array(g_c_w + g_c_wh)
        Ψ_1_G = onp.array(g_c_xw)
        Ψ_2_G = onp.array(g_c_ww)

    elif growth_variable == "G":
        Γ_0_G = onp.array(η_0_g + g_g_h)
        Γ_1_G = onp.array(g_g_x + g_g_xh)
        Γ_2_G = onp.array(h**2 / 2 * g_g_x)
        Γ_3_G = onp.array(g_g_xx)
        Ψ_0_G = onp.array(g_g_w + g_g_wh)
        Ψ_1_G = onp.array(g_g_xw)
        Ψ_2_G = onp.array(g_g_ww)

    elif growth_variable == "S":
        Γ_0_G = 0
        Γ_1_G = 0
        Γ_2_G = 0
        Γ_3_G = 0
        Ψ_0_G = 0
        Ψ_1_G = 0
        Ψ_2_G = 0

    else:
        raise ValueError(f"growth variable must be 'C', 'G', or 'S'. You put '{growth_variable}'.")

    Γ_0_SG = Γ_0_G + onp.array(onp.log(δ) - ρ * (η_0_c + g_c_h) + \
                k1 * (S_1_vmc.T @ ψ_h + (1 - 1/λ) * η_1_vmc + g_c_h) + k2 * d_2)

    Γ_1_SG = Γ_1_G + onp.array( - ρ * (g_c_x + g_c_xh) + \
                k1 * (S_1_vmc.T @ ψ_x + g_c_x - S_1_vmc.T / λ) - k2 * E_w.T@b_2)

    Γ_2_SG = Γ_2_G + onp.array(h**2 / 2 * (- ρ * g_c_x))

    Γ_3_SG = Γ_3_G + onp.array(- ρ * g_c_xx)

    Ψ_0_SG = Ψ_0_G + onp.array( - ρ * (g_c_w + g_c_wh) + \
                k1 * (S_1_vmc.T @ ψ_w + g_c_w) + k2 * a_2)

    Ψ_1_SG = Ψ_1_G + onp.array(- ρ * g_c_xw + vec(k2 * b_2).T)

    Ψ_2_SG = Ψ_2_G + onp.array(- ρ * g_c_ww + k2 * c_2)

    ψ_x = onp.array(ψ_x)
    ψ_w = onp.array(ψ_w)
    ψ_h = onp.array(ψ_h)

    ψ_xx = onp.array(ψ_xx)
    ψ_xw = onp.array(ψ_xw)
    ψ_xh = onp.array(ψ_xh)

    ψ_ww = onp.array(ψ_ww)
    ψ_wh = onp.array(ψ_wh)

    ψ_hh = onp.array(ψ_hh)

    # print(ψ_x, ψ_w, ψ_h, ψ_xx, ψ_xw, ψ_xh, ψ_ww, ψ_wh, ψ_hh)

    return ψ_x, ψ_w, ψ_h, ψ_xx, ψ_xw, ψ_xh, ψ_ww, ψ_wh, ψ_hh, \
        Γ_0_G, Γ_1_G, Γ_2_G, Γ_3_G, Ψ_0_G, Ψ_1_G, Ψ_2_G, \
        Γ_0_SG, Γ_1_SG, Γ_2_SG, Γ_3_SG, Ψ_0_SG, Ψ_1_SG, Ψ_2_SG


def simulation(ψ_x, ψ_w, ψ_h, ψ_xx, ψ_xw, ψ_xh, ψ_ww, ψ_wh, ψ_hh, T = 79):

    n,k = ψ_w.shape

    X_1 = onp.zeros((n, T * 12))
    X_2 = onp.zeros((n, T * 12))
    x = onp.zeros(T*12)
    σ_squared = onp.zeros(T*12)

    Ws = onp.random.multivariate_normal(onp.zeros(k), onp.eye(k), T * 12)
    πs = onp.zeros(T * 12)

    X_1[:,0] = ψ_w @ Ws[0] + ψ_h[:,0]
    X_2[:,0] = (ψ_wh @ Ws[0]) + ψ_hh[:,0]
    X_2[0,0] = Ws[0].T @ mat(ψ_ww[0], (k,k)) @ Ws[0]
    X_2[1,0] = Ws[0].T @ mat(ψ_ww[1], (k,k)) @ Ws[0]

    x[0] = ϕ_e * σ * Ws[0,0]
    σ_squared[0] = max(σ**2 + σ_w * Ws[0,1], 0)

    for i in range(1, T * 12):
        X_1[:,i] = ψ_x @ X_1[:,i-1] + ψ_w @ Ws[i] + ψ_h[:,0]
        X_2[:,i] = ψ_x @ X_2[:,i-1] + (ψ_xh @ X_1[:,i-1] + ψ_hh)[:,0] + (ψ_wh @ Ws[i])
        X_2[0,i] = X_1[:,i-1].T @ ψ_xx[0].reshape((n,n)) @ X_1[:,i-1] + \
                X_1[:,i-1].T @ ψ_xw[0].reshape((n,k)) @ Ws[i] + Ws[i].T @ ψ_ww[0].reshape((k,k)) @ Ws[i]
        X_2[1,i] = X_1[:,i-1].T @ ψ_xx[1].reshape((n,n)) @ X_1[:,i-1] + \
                X_1[:,i-1].T @ ψ_xw[1].reshape((n,k)) @ Ws[i] + Ws[i].T @ ψ_ww[1].reshape((k,k)) @ Ws[i]

        x[i] = α * x[i-1] + ϕ_e * onp.sqrt(σ_squared[i-1]) * Ws[i,0]
        σ_squared[i] = max(σ**2 + ν_1 * (σ_squared[i-1] - σ**2) + σ_w * Ws[i,1] * onp.sqrt(σ_squared[i-1]),0)

    return X_1, X_2, x, σ_squared

# def simulation_π(amf, ψ_x, ψ_w, ψ_h, ψ_xx, ψ_xw, ψ_xh, ψ_ww, ψ_wh, ψ_hh, T = 79, find_π = True):

#     ϕ_e = 0.044
#     ν_1 = 0.987
#     σ = 0.0078
#     σ_w = 0.23 * 1e-5
#     α = 0.979

#     X_1 = onp.zeros((n, T * 12))
#     X_2 = onp.zeros((n, T * 12))
#     x = onp.zeros(T*12)
#     σ_squared = onp.zeros(T*12)

#     Ws = onp.random.multivariate_normal(onp.zeros(m), onp.eye(m), T * 12)
#     πs = onp.zeros(T * 12)

#     X_1[:,0] = ψ_w @ Ws[0] + ψ_h[:,0]
#     X_2[:,0] = (ψ_wh @ Ws[0])[:,0] + ψ_hh[:,0]
#     X_2[0,0] = Ws[0].T @ ψ_ww[0] @ Ws[0]
#     X_2[1,0] = Ws[0].T @ ψ_ww[1] @ Ws[0]
#     x[0] = ϕ_e * σ * Ws[0,0]
#     σ_squared[0] = σ**2 + σ_w * Ws[0,1]

#     _, _, πs[0] = π_t_decomp(amf, X_1[:,0], X_2[:,0], N=15*12)

#     for i in range(1, T * 12):
#         X_1[:,i] = ψ_x @ X_1[:,i-1] + ψ_w @ Ws[i] + ψ_h[:,0]
#         X_2[:,i] = ψ_x @ X_2[:,i-1] + (ψ_xh @ X_1[:,i-1] + ψ_hh)[:,0] + (ψ_wh @ Ws[i])[:,0]
#         X_2[0,i] = X_1[:,i-1].T @ ψ_xx[0] @ X_1[:,i-1] + X_1[:,i-1].T @ ψ_xw[0] @ Ws[i] + Ws[i].T @ ψ_ww[0] @ Ws[i]
#         X_2[1,i] = X_1[:,i-1].T @ ψ_xx[1] @ X_1[:,i-1] + X_1[:,i-1].T @ ψ_xw[1] @ Ws[i] + Ws[i].T @ ψ_ww[1] @ Ws[i]
#         if find_π:
#             _, _, πs[i] = π_t_decomp(amf, X_1[:,i], X_2[:,i], N = 15*12)
#         x[i] = α * x[i-1] + ϕ_e * onp.sqrt(σ_squared[i-1]) * Ws[i,0]
#         σ_squared[i] = σ**2 + ν_1 * (σ_squared[i-1] - σ**2) + σ_w * Ws[i,1]

#     if find_π:
#         dat100 = loadmat("BY_first_order_gamma_10.mat")
#         A_0m = dat100['A_0m']
#         A_1m = dat100['A_1m']
#         A_2m = dat100['A_2m']
#         π_BY = A_0m + A_1m * x + A_2m * σ_squared
#         return onp.log(πs), π_BY[0]
#     else:
#         return X_1, X_2

# def π_t_decomp(amf, X_1_t, X_2_t, N = num_periods):
#     # This function is the same as the above function but will allow for the decomposition
#     # of date t price-dividend ratio into a tail term and the calculated terms.

#     # Ensure that the switching point between calculated coefficients and limiting coefficients is okay
#     if N > len(amf.P_t_bar_path):
#         N = len(amf.P_t_bar_path)
#     π = 0
#     # Iterate over the N first calculated terms
#     for i in range(N):
#         P_i = amf.P_t_bar_path[i]
#         π_current = onp.exp(P_i[0] + P_i[1] @ X_1_t + P_i[2] @ X_2_t + P_i[3] @ onp.kron(X_1_t, X_1_t))
#         π += π_current
#     π_1 = π
#     π_2 = onp.exp(P_i[0] + Γ_0_bar_diff + Γ_1_bar @ X_1_t + \
#                   Γ_2_bar @ X_2_t + Γ_3_bar @ onp.kron(X_1_t, X_1_t)) / (1 - onp.exp(Γ_0_bar_diff))
#     π_total = π_1 + π_2
#     return π_1, π_2, π_total

def find_limiting_vector(triss, 𝒫, x1, x2, perturbed_model_params):
    Γ_0, Γ_1, Γ_2, Γ_3, Ψ_0, Ψ_1, Ψ_2 = 𝒫

    ψ_h = perturbed_model_params['ψ_q']
    ψ_x = perturbed_model_params['ψ_x']
    ψ_w = perturbed_model_params['ψ_w']
    ψ_hh = perturbed_model_params['ψ_qq']
    ψ_xh = perturbed_model_params['ψ_xq']
    ψ_x = perturbed_model_params['ψ_x']
    ψ_xx = perturbed_model_params['ψ_xx']
    ψ_wh = perturbed_model_params['ψ_wq']
    ψ_xw = perturbed_model_params['ψ_xw']
    ψ_ww = perturbed_model_params['ψ_ww']

    Γ_2_bar = onp.linalg.solve((onp.eye(len(triss.Θ_22)) - triss.Θ_22).T, Γ_2.T).T

    Θ_10 = triss.Θ_10
    Θ_11 = triss.Θ_11
    Θ_23 = triss.Θ_23
    Λ_10 = triss.Λ_10
    Λ_21 = triss.Λ_21
    Λ_22 = triss.Λ_22

    @jit
    def iteration(Γ_3_bar):
        n, k = Θ_10.shape[0], Λ_10.shape[1]

        temp = onp.hstack([onp.kron(Λ_10, Θ_11[:, [j]]) for j in range(n)])
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
    maxiters = 10000
    tol = 1e-10
    while diff > tol and iters < maxiters:
        iters += 1
        Γ_3_bar_new = iteration(Γ_3_bar)
        # diff = onp.abs(onp.max(onp.max((Γ_3_bar - Γ_3_bar_new)/Γ_3_bar)))
        diff = onp.abs(onp.max(onp.max(Γ_3_bar - Γ_3_bar_new)))
        Γ_3_bar = onp.copy(Γ_3_bar_new)
    if diff > tol:
        print(diff)
        print(f"Did not converge after {maxiters} iterations.")
    else:
        print(f"Converged after {iters} iterations.")

    n, k = Θ_10.shape[0], Λ_10.shape[1]
    temp = onp.hstack([onp.kron(Λ_10, Θ_11[:, [j]]) for j in range(n)])
    Ψ_1_tilde = Γ_2_bar @ Λ_21 + Γ_3_bar @ (onp.kron(Θ_11, Λ_10) + temp)
    Ψ_2_tilde = Γ_2_bar @ Λ_22 + Γ_3_bar @ onp.kron(Λ_10, Λ_10)
    C1 = Γ_2_bar @ triss.Θ_21 + Γ_3_bar @ (onp.kron(Θ_10, Θ_11) + onp.kron(Θ_11, Θ_10))
    Σ_inv = onp.eye(k) - sym(mat(2 * (Ψ_2_tilde + Ψ_2), (k, k)))
    Σ = onp.linalg.inv(Σ_inv)
    A = Σ @ mat(Ψ_1_tilde + Ψ_1, (k, n))
    C2 = Γ_2_bar @ triss.Λ_20 + Γ_3_bar @ (onp.kron(Θ_10, Λ_10) + onp.kron(Λ_10, Θ_10))

    Γ_1_bar = onp.linalg.solve((onp.eye(len(Θ_11)) - Θ_11 - Λ_10 @ A).T, (Γ_1 + C1 + Ψ_0@A + C2@A).T).T

    Ψ_0_tilde = Γ_1_bar @ Λ_10 + C2

    C1 = - 1 / 2 * onp.log(onp.linalg.det(Σ_inv)) + \
                1 / 2 * (Ψ_0 + Ψ_0_tilde) @ Σ @ (Ψ_0 + Ψ_0_tilde).T
    C2 = Γ_1_bar @ Θ_10 + Γ_2_bar @ triss.Θ_20 + Γ_3_bar @ onp.kron(Θ_10, Θ_10)
    η = Γ_0 + C1 + C2

    const = Γ_0 + Γ_1_bar @ ψ_h + Γ_2_bar @ ψ_hh + Γ_3_bar @ (onp.kron(ψ_h, ψ_h))
    x1_term = Γ_1 + Γ_1_bar @ (ψ_x - onp.eye(n)) + 2 * Γ_2_bar @ ψ_xh +\
                    Γ_3_bar @ (onp.kron(ψ_x, ψ_h) + onp.kron(ψ_h, ψ_x))
    x2_term = Γ_2 + Γ_2_bar @ (ψ_x - onp.eye(n))
    x1ox1_term = Γ_3 + Γ_2_bar @ ψ_xx + Γ_3_bar @ (onp.kron(ψ_x, ψ_x) - onp.eye(n**2))

    A = Ψ_0 + Γ_1_bar@ψ_w + x1.T@mat(2*Γ_2_bar@ψ_xw, (k,n)).T + 2*Γ_2_bar@ψ_wh +\
        x1.T@(mat(Γ_3_bar@onp.kron(ψ_x,ψ_w), (k,n)).T + mat(Γ_3_bar@onp.kron(ψ_w,ψ_x), (n,k))) +\
        Γ_3_bar @ (onp.kron(ψ_w, ψ_h) + onp.kron(ψ_h, ψ_w)) + x1.T@mat(Ψ_1, (k,n)).T

    B = Ψ_2 + Γ_2_bar @ ψ_ww + Γ_3_bar @ onp.kron(ψ_w, ψ_w)
    limits = onp.linalg.inv(onp.eye(k) - sym(mat(2*B, (k,k))))@A.T * \
        onp.exp(const + x1_term@x1 + x2_term@x2 + x1ox1_term@onp.kron(x1, x1))

    return limits
