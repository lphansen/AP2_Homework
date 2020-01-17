# Import necessary packages
from utils import *
import matplotlib
import matplotlib.pyplot as plt
import time
from amf import *
from tri_ss import *
import pprint
from numba import jit, njit
from scipy.io import loadmat
import time
import pandas as pd
import warnings
from scipy.stats import norm
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot
#     import plotly.plotly as py
    import plotly.io as pio
except ImportError:
    print("Installing plotly. This may take a while.")
    from pip._internal import main as pipmain
    pipmain(['install', 'plotly'])
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot
#     import plotly.plotly as py
    import plotly.io as pio

ρs = [2./3, 1.00001, 3./2]
γs = [1.00001, 3, 5, 7, 10]
keys = ["low", "mid", "hi"]
dom = onp.arange(150*3)[1:]/3.

def get_elasticities(amf_g, amf_sg, α_h, x, num_periods):
    ϵ_g = onp.zeros(num_periods - 1)
    ϵ_p = onp.zeros(num_periods - 1)

    for t in range(num_periods - 1):
        ϵ_g[t] = amf_g.ε(x, t + 1, α_h)
        ϵ_p[t] = amf_g.ε(x, t + 1, α_h) - amf_sg.ε(x, t + 1, α_h)

    return ϵ_g, ϵ_p

def make_plot_data(γ, ρ, growth_measure, δ):
    ψ_x, ψ_w, ψ_h, ψ_xx, ψ_xw, ψ_xh, ψ_ww, ψ_wh, ψ_hh, \
        Γ_0_g, Γ_1_g, Γ_2_g, Γ_3_g, Ψ_0_g, Ψ_1_g, Ψ_2_g, \
        Γ_0_sg, Γ_1_sg, Γ_2_sg, Γ_3_sg, Ψ_0_sg, Ψ_1_sg, Ψ_2_sg = derivs_BY(growth_measure, ρ, γ, δ)

    𝒫_sg = (Γ_0_sg, Γ_1_sg, Γ_2_sg, Γ_3_sg, Ψ_0_sg, Ψ_1_sg, Ψ_2_sg)
    𝒫_g = (Γ_0_g, Γ_1_g, Γ_2_g, Γ_3_g, Ψ_0_g, Ψ_1_g, Ψ_2_g)

    perturbed_model_params = {
            'ψ_q': ψ_h,
            'ψ_x': ψ_x,
            'ψ_w': ψ_w,
            'ψ_qq': ψ_hh,
            'ψ_xq': ψ_xh,
            'ψ_x': ψ_x,
            'ψ_xx': ψ_xx,
            'ψ_wq': ψ_wh,
            'ψ_xw': ψ_xw,
            'ψ_ww': ψ_ww
            }

    triss = map_perturbed_model_to_tri_ss(perturbed_model_params)
    amf_sg = Amf(𝒫_sg, triss)
    amf_g = Amf(𝒫_g, triss)

    num_periods = 150*3
    amf_sg.iterate(num_periods)
    amf_g.iterate(num_periods)

    σ_w = 0.23 * 1e-5
    ν_1 = 0.987
    σ = 0.0078
    std_dev = onp.sqrt(σ_w**2 / (1 - ν_1**2))
    offset = norm.ppf(.9)

    x = onp.array([[0,0]])
    x_low = onp.array([[0, -std_dev * offset]])
    x_high = onp.array([[0, std_dev * offset]])

    scaler_low = np.sqrt(-std_dev * offset + σ**2)/σ
    scaler_high = np.sqrt(std_dev * offset + σ**2)/σ

    ϵ_g1 = {}
    ϵ_p1 = {}
    ϵ_g2 = {}
    ϵ_p2 = {}
    ϵ_g3 = {}
    ϵ_p3 = {}
    ϵ_g4 = {}
    ϵ_p4 = {}

    # Calculate the elasticities at the mean of stochastic volatility
    ϵ_g1['mid'], ϵ_p1['mid'] = get_elasticities(amf_g, amf_sg, \
                                                onp.array([1, 0, 0, 0]), x, num_periods)
    ϵ_g2['mid'], ϵ_p2['mid'] = get_elasticities(amf_g, amf_sg, \
                                                -onp.array([0, 1, 0, 0]), x, num_periods)
    ϵ_g3['mid'], ϵ_p3['mid'] = get_elasticities(amf_g, amf_sg, \
                                                onp.array([0, 0, 1, 0]), x, num_periods)
    ϵ_g4['mid'], ϵ_p4['mid'] = get_elasticities(amf_g, amf_sg, \
                                                onp.array([0, 0, 0, 1]), x, num_periods)

    # Calculate the elasticities at the .25 quartile
    ϵ_g1['low'], ϵ_p1['low'] = get_elasticities(amf_g, amf_sg, \
                                                onp.array([1, 0, 0, 0])*scaler_low, x_low, num_periods)
    ϵ_g2['low'], ϵ_p2['low'] = get_elasticities(amf_g, amf_sg, \
                                                -onp.array([0, 1, 0, 0])*scaler_low, x_low, num_periods)
    ϵ_g3['low'], ϵ_p3['low'] = get_elasticities(amf_g, amf_sg, \
                                                onp.array([0, 0, 1, 0])*scaler_low, x_low, num_periods)
    ϵ_g4['low'], ϵ_p4['low'] = get_elasticities(amf_g, amf_sg, \
                                                onp.array([0, 0, 0, 1])*scaler_low, x_low, num_periods)

    # Calculate the elasticities at the .75 quartile
    ϵ_g1['hi'], ϵ_p1['hi'] = get_elasticities(amf_g, amf_sg, \
                                                onp.array([1, 0, 0, 0])*scaler_high, x_high, num_periods)
    ϵ_g2['hi'], ϵ_p2['hi'] = get_elasticities(amf_g, amf_sg, \
                                                -onp.array([0, 1, 0, 0])*scaler_high, x_high, num_periods)
    ϵ_g3['hi'], ϵ_p3['hi'] = get_elasticities(amf_g, amf_sg, \
                                                onp.array([0, 0, 1, 0])*scaler_high, x_high, num_periods)
    ϵ_g4['hi'], ϵ_p4['hi'] = get_elasticities(amf_g, amf_sg, \
                                                onp.array([0, 0, 0, 1])*scaler_high, x_high, num_periods)

    return ϵ_g1, ϵ_g2, ϵ_g3, ϵ_g4, ϵ_p1, ϵ_p2, ϵ_p3, ϵ_p4

def slide_prices(fix, slide, β):
    if slide == "both":
        slide_vars = γs
        label_name = 'γ = ρ'
    elif slide == "γ":
        slide_vars = γs
        label_name = slide
    elif slide == "ρ":
        slide_vars = ρs
        label_name = slide
    fig = make_subplots(3, 1, print_grid = False)

    mins = [0,0,0]
    maxs = [0,0,0]

    for i, slide_var in enumerate(slide_vars):
        if slide == "γ":
            ϵ_g1, ϵ_g2, ϵ_g3, ϵ_g4, ϵ_p1, ϵ_p2, ϵ_p3, ϵ_p4 = make_plot_data(slide_var, fix, "C", β)
        elif slide == "ρ":
            ϵ_g1, ϵ_g2, ϵ_g3, ϵ_g4, ϵ_p1, ϵ_p2, ϵ_p3, ϵ_p4 = make_plot_data(fix, slide_var, "C", β)
        elif slide == "both":
            ϵ_g1, ϵ_g2, ϵ_g3, ϵ_g4, ϵ_p1, ϵ_p2, ϵ_p3, ϵ_p4 = make_plot_data(slide_var, slide_var, "C", β)

        scale = 1.2 * onp.sqrt(12)

        mins[0] = min(mins[0], onp.min(ϵ_p3['mid']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_p1['mid']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_p2['mid']) * scale)

        mins[0] = min(mins[0], onp.min(ϵ_p3['hi']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_p1['hi']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_p2['hi']) * scale)

        mins[0] = min(mins[0], onp.min(ϵ_p3['low']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_p1['low']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_p2['low']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_p3['mid']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_p1['mid']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_p2['mid']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_p3['hi']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_p1['hi']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_p2['hi']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_p3['low']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_p1['low']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_p2['low']) * scale)

        fig.add_scatter(x = dom, y = ϵ_p3['mid'] * onp.sqrt(12), row = 1, col = 1, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))
        fig.add_scatter(x = dom, y = ϵ_p1['mid'] * onp.sqrt(12), row = 2, col = 1, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))
        fig.add_scatter(x = dom, y = ϵ_p2['mid'] * onp.sqrt(12), row = 3, col = 1, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))

        fig.add_scatter(x = dom, y = ϵ_p3['hi'] * onp.sqrt(12), row = 1, col = 1, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = ϵ_p3['low'] * onp.sqrt(12), row = 1, col = 1, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

        fig.add_scatter(x = dom, y = ϵ_p1['hi'] * onp.sqrt(12), row = 2, col = 1, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = ϵ_p1['low'] * onp.sqrt(12), row = 2, col = 1, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

        fig.add_scatter(x = dom, y = ϵ_p2['hi'] * onp.sqrt(12), row = 3, col = 1, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = ϵ_p2['low'] * onp.sqrt(12), row = 3, col = 1, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

    steps = []
    for i in range(len(slide_vars)):
        step = dict(
            method = 'restyle',
            args = ['visible', ['legendonly'] * len(fig.data)],
            label = f'{label_name} = '+'{}'.format(round(slide_vars[i], 2))
        )
        for j in range(9):
            step['args'][1][i * 9 + j] = True
        steps.append(step)

    sliders = [dict(
        steps = steps,
        pad = dict(t=30)
    )]

    fig.layout.sliders = sliders

    fig['layout'].update(height=800, width=700,
                     title=f"Shock Price Elasticities holding γ fixed at {fix}", showlegend = False)

    fig['layout']['yaxis1'].update(title='Consumption', range = [mins[0], maxs[0]])
    fig['layout']['yaxis2'].update(title='Long Run Growth', range=[mins[1], maxs[1]])
    fig['layout']['yaxis3'].update(title='Stochastic Volatility', range = [mins[2], maxs[2]])

    fig['layout']['xaxis1'].update(title='Maturity (quarters)')
    fig['layout']['xaxis2'].update(title='Maturity (quarters)')
    fig['layout']['xaxis3'].update(title='Maturity (quarters)')


    return fig

def slide_costs(fix, slide, growth_measure, β):
    if slide == "both":
        slide_vars = γs
        label_name = 'γ = ρ'
    elif slide == "γ":
        slide_vars = γs
        label_name = slide
    elif slide == "ρ":
        slide_vars = ρs
        label_name = slide
    fig = make_subplots(4, 1, print_grid = False)

    mins = [0,0,0,0]
    maxs = [0,0,0,0]

    for i, slide_var in enumerate(slide_vars):
        if slide == "γ":
            ϵ_g1, ϵ_g2, ϵ_g3, ϵ_g4, ϵ_p1, ϵ_p2, ϵ_p3, ϵ_p4 = make_plot_data(slide_var, fix, growth_measure, β)
        elif slide == "ρ":
            ϵ_g1, ϵ_g2, ϵ_g3, ϵ_g4, ϵ_p1, ϵ_p2, ϵ_p3, ϵ_p4 = make_plot_data(fix, slide_var, growth_measure, β)
        elif slide == "both":
            ϵ_g1, ϵ_g2, ϵ_g3, ϵ_g4, ϵ_p1, ϵ_p2, ϵ_p3, ϵ_p4 = make_plot_data(slide_var, slide_var, growth_measure, β)

        scale = 1.2 * onp.sqrt(12)

        mins[0] = min(mins[0], onp.min(ϵ_g3['mid']-ϵ_p3['mid']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_g1['mid']-ϵ_p1['mid']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_g2['mid']-ϵ_p2['mid']) * scale)
        mins[3] = min(mins[3], onp.min(ϵ_g4['mid']-ϵ_p4['mid']) * scale)

        mins[0] = min(mins[0], onp.min(ϵ_g3['hi']-ϵ_p3['hi']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_g1['hi']-ϵ_p1['hi']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_g2['hi']-ϵ_p2['hi']) * scale)
        mins[3] = min(mins[3], onp.min(ϵ_g4['hi']-ϵ_p4['hi']) * scale)

        mins[0] = min(mins[0], onp.min(ϵ_g3['low']-ϵ_p3['low']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_g1['low']-ϵ_p1['low']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_g2['low']-ϵ_p2['low']) * scale)
        mins[3] = min(mins[3], onp.min(ϵ_g4['low']-ϵ_p4['low']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_g3['mid']-ϵ_p3['mid']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_g1['mid']-ϵ_p1['mid']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_g2['mid']-ϵ_p2['mid']) * scale)
        maxs[3] = max(maxs[3], onp.max(ϵ_g4['mid']-ϵ_p4['mid']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_g3['hi']-ϵ_p3['hi']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_g1['hi']-ϵ_p1['hi']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_g2['hi']-ϵ_p2['hi']) * scale)
        maxs[3] = max(maxs[3], onp.max(ϵ_g4['hi']-ϵ_p4['hi']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_g3['low']-ϵ_p3['low']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_g1['low']-ϵ_p1['low']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_g2['low']-ϵ_p2['low']) * scale)
        maxs[3] = max(maxs[3], onp.max(ϵ_g4['low']-ϵ_p4['low']) * scale)

        fig.add_scatter(x = dom, y = (ϵ_g3['mid']-ϵ_p3['mid']) * onp.sqrt(12), row = 1, col = 1, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))
        fig.add_scatter(x = dom, y = (ϵ_g1['mid']-ϵ_p1['mid']) * onp.sqrt(12), row = 2, col = 1, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))
        fig.add_scatter(x = dom, y = (ϵ_g2['mid']-ϵ_p2['mid']) * onp.sqrt(12), row = 3, col = 1, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))
        fig.add_scatter(x = dom, y = (ϵ_g4['mid']-ϵ_p4['mid']) * onp.sqrt(12), row = 4, col = 1, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))

        fig.add_scatter(x = dom, y = (ϵ_g3['hi']-ϵ_p3['hi']) * onp.sqrt(12), row = 1, col = 1, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = (ϵ_g3['low']-ϵ_p3['low']) * onp.sqrt(12), row = 1, col = 1, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

        fig.add_scatter(x = dom, y = (ϵ_g1['hi']-ϵ_p1['hi']) * onp.sqrt(12), row = 2, col = 1, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = (ϵ_g1['low']-ϵ_p1['low']) * onp.sqrt(12), row = 2, col = 1, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

        fig.add_scatter(x = dom, y = (ϵ_g2['hi']-ϵ_p2['hi']) * onp.sqrt(12), row = 3, col = 1, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = (ϵ_g2['low']-ϵ_p2['low']) * onp.sqrt(12), row = 3, col = 1, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

        fig.add_scatter(x = dom, y = (ϵ_g4['hi']-ϵ_p4['hi']) * onp.sqrt(12), row = 4, col = 1, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = (ϵ_g4['low']-ϵ_p4['low']) * onp.sqrt(12), row = 4, col = 1, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

    steps = []
    for i in range(len(slide_vars)):
        step = dict(
            method = 'restyle',
            args = ['visible', ['legendonly'] * len(fig.data)],
            label = f'{label_name} = '+'{}'.format(round(slide_vars[i], 2))
        )
        for j in range(12):
            step['args'][1][i * 12 + j] = True
        steps.append(step)

    sliders = [dict(
        steps = steps,
        pad = dict(t=30)
    )]

    fig.layout.sliders = sliders

    if growth_measure == "C":
        title_append = "Consumption Growth"
    elif growth_measure == "G":
        title_append = "Dividend Growth"

    fig['layout'].update(height=800, width=700,
                     title=f"Shock Cost Elasticities holding γ fixed at {fix}, {title_append}", showlegend = False)

    fig['layout']['yaxis1'].update(title='Consumption Growth', range = [mins[0], maxs[0]])
    fig['layout']['yaxis2'].update(title='Long Run Growth', range=[mins[1], maxs[1]])
    fig['layout']['yaxis3'].update(title='Stochastic Volatility', range = [mins[2], maxs[2]])
    fig['layout']['yaxis4'].update(title='Dividend Growth', range = [mins[3], maxs[3]])

#     fig['layout']['xaxis1'].update(title='Maturity (quarters)')
#     fig['layout']['xaxis2'].update(title='Maturity (quarters)')
#     fig['layout']['xaxis3'].update(title='Maturity (quarters)')
    fig['layout']['xaxis4'].update(title='Maturity (quarters)')
    return fig

def make_price_preference_comparison(fixed_rho = 1.00001, β = 0.998):
    slide_vars = γs
    label_name = "γ"

    fig = make_subplots(3, 2, print_grid = False, subplot_titles=(f"ρ fixed at {fixed_rho}", "ρ = γ"))

    mins = [0,0,0]
    maxs = [0,0,0]

    for i, slide_var in enumerate(slide_vars):
        ϵ_g1, ϵ_g2, ϵ_g3, ϵ_g4, ϵ_p1, ϵ_p2, ϵ_p3, ϵ_p4 = make_plot_data(slide_var, fixed_rho, "C", β)

        scale = 1.2 * onp.sqrt(12)

        mins[0] = min(mins[0], onp.min(ϵ_p3['mid']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_p1['mid']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_p2['mid']) * scale)

        mins[0] = min(mins[0], onp.min(ϵ_p3['hi']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_p1['hi']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_p2['hi']) * scale)

        mins[0] = min(mins[0], onp.min(ϵ_p3['low']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_p1['low']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_p2['low']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_p3['mid']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_p1['mid']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_p2['mid']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_p3['hi']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_p1['hi']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_p2['hi']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_p3['low']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_p1['low']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_p2['low']) * scale)

        fig.add_scatter(x = dom, y = ϵ_p3['mid'] * onp.sqrt(12), row = 1, col = 1, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))
        fig.add_scatter(x = dom, y = ϵ_p1['mid'] * onp.sqrt(12), row = 2, col = 1, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))
        fig.add_scatter(x = dom, y = ϵ_p2['mid'] * onp.sqrt(12), row = 3, col = 1, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))

        fig.add_scatter(x = dom, y = ϵ_p3['hi'] * onp.sqrt(12), row = 1, col = 1, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = ϵ_p3['low'] * onp.sqrt(12), row = 1, col = 1, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

        fig.add_scatter(x = dom, y = ϵ_p1['hi'] * onp.sqrt(12), row = 2, col = 1, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = ϵ_p1['low'] * onp.sqrt(12), row = 2, col = 1, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

        fig.add_scatter(x = dom, y = ϵ_p2['hi'] * onp.sqrt(12), row = 3, col = 1, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = ϵ_p2['low'] * onp.sqrt(12), row = 3, col = 1, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

        ϵ_g1, ϵ_g2, ϵ_g3, ϵ_g4, ϵ_p1, ϵ_p2, ϵ_p3, ϵ_p4 = make_plot_data(slide_var, slide_var, "C", β)

        scale = 1.2 * onp.sqrt(12)

        mins[0] = min(mins[0], onp.min(ϵ_p3['mid']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_p1['mid']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_p2['mid']) * scale)

        mins[0] = min(mins[0], onp.min(ϵ_p3['hi']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_p1['hi']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_p2['hi']) * scale)

        mins[0] = min(mins[0], onp.min(ϵ_p3['low']) * scale)
        mins[1] = min(mins[1], onp.min(ϵ_p1['low']) * scale)
        mins[2] = min(mins[2], onp.min(ϵ_p2['low']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_p3['mid']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_p1['mid']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_p2['mid']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_p3['hi']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_p1['hi']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_p2['hi']) * scale)

        maxs[0] = max(maxs[0], onp.max(ϵ_p3['low']) * scale)
        maxs[1] = max(maxs[1], onp.max(ϵ_p1['low']) * scale)
        maxs[2] = max(maxs[2], onp.max(ϵ_p2['low']) * scale)

        fig.add_scatter(x = dom, y = ϵ_p3['mid'] * onp.sqrt(12), row = 1, col = 2, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))
        fig.add_scatter(x = dom, y = ϵ_p1['mid'] * onp.sqrt(12), row = 2, col = 2, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))
        fig.add_scatter(x = dom, y = ϵ_p2['mid'] * onp.sqrt(12), row = 3, col = 2, visible = i == 0,
                        name = "mean", line = dict(color = ("blue")))

        fig.add_scatter(x = dom, y = ϵ_p3['hi'] * onp.sqrt(12), row = 1, col = 2, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = ϵ_p3['low'] * onp.sqrt(12), row = 1, col = 2, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

        fig.add_scatter(x = dom, y = ϵ_p1['hi'] * onp.sqrt(12), row = 2, col = 2, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = ϵ_p1['low'] * onp.sqrt(12), row = 2, col = 2, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

        fig.add_scatter(x = dom, y = ϵ_p2['hi'] * onp.sqrt(12), row = 3, col = 2, visible = i == 0,
                        name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
        fig.add_scatter(x = dom, y = ϵ_p2['low'] * onp.sqrt(12), row = 3, col = 2, visible = i == 0,
                        name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')


    steps = []
    for i in range(len(slide_vars)):
        step = dict(
            method = 'restyle',
            args = ['visible', ['legendonly'] * len(fig.data)],
            label = f'{label_name} = '+'{}'.format(round(slide_vars[i], 2))
        )
        for j in range(18):
            step['args'][1][i * 18 + j] = True
        steps.append(step)

    sliders = [dict(
        steps = steps,
        pad = dict(t=30)
    )]

    fig.layout.sliders = sliders

    fig['layout'].update(height=800, width=1000, showlegend = False, title="Shock Price Elasticities")

    fig['layout']['yaxis1'].update(title='Consumption', range = [mins[0], maxs[0]])
    fig['layout']['yaxis3'].update(title='Long Run Growth', range=[mins[1], maxs[1]])
    fig['layout']['yaxis5'].update(title='Stochastic Volatility', range = [mins[2], maxs[2]])
    fig['layout']['yaxis2'].update(range = [mins[0], maxs[0]])
    fig['layout']['yaxis4'].update(range=[mins[1], maxs[1]])
    fig['layout']['yaxis6'].update(range = [mins[2], maxs[2]])

    fig['layout']['xaxis5'].update(title='Maturity (quarters)')
    fig['layout']['xaxis6'].update(title='Maturity (quarters)')

    return fig

def view_exposure_elasticities(growth_variable, β):

    if growth_variable == "G":
        title = "Dividend Growth"
    elif growth_variable == "C":
        title = "Consumption Growth"
    else:
        raise valueError(f"Growth variable must be 'C' or 'G'. You put '{growth_variable}'.")

    ϵ_g1, ϵ_g2, ϵ_g3, ϵ_g4, ϵ_p1, ϵ_p2, ϵ_p3, ϵ_p4 = make_plot_data(10, 1.00001, growth_variable, β)

    fig = make_subplots(3, 1, print_grid = False)

    fig.add_scatter(x = dom, y = ϵ_g3['mid'] * onp.sqrt(12), row = 1, col = 1,
                    name = "mean", line = dict(color = ("blue")))
    fig.add_scatter(x = dom, y = ϵ_g1['mid'] * onp.sqrt(12), row = 2, col = 1,
                    name = "mean", line = dict(color = ("blue")))
    fig.add_scatter(x = dom, y = ϵ_g2['mid'] * onp.sqrt(12), row = 3, col = 1,
                    name = "mean", line = dict(color = ("blue")))

    fig.add_scatter(x = dom, y = ϵ_g3['hi'] * onp.sqrt(12), row = 1, col = 1,
                    name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
    fig.add_scatter(x = dom, y = ϵ_g3['low'] * onp.sqrt(12), row = 1, col = 1,
                    name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

    fig.add_scatter(x = dom, y = ϵ_g1['hi'] * onp.sqrt(12), row = 2, col = 1,
                    name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
    fig.add_scatter(x = dom, y = ϵ_g1['low'] * onp.sqrt(12), row = 2, col = 1,
                    name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

    fig.add_scatter(x = dom, y = ϵ_g2['hi'] * onp.sqrt(12), row = 3, col = 1,
                    name = "0.9 decile", line = dict(color = ("blue"), dash = 'dash'))
    fig.add_scatter(x = dom, y = ϵ_g2['low'] * onp.sqrt(12), row = 3, col = 1,
                    name = "0.1 decile", line = dict(color = ("blue"), dash = 'dash'), fill = 'tonexty')

    fig['layout'].update(height=800, width=700,
                     title=f"Shock Exposure Elasticities, {title}", showlegend = False)

    fig['layout']['yaxis1'].update(title='Consumption')#, range=[0, .1])
    fig['layout']['yaxis2'].update(title='Long Run Growth')#, range=[0, .1])
    fig['layout']['yaxis3'].update(title='Stochastic Volatility')#, range=[-3e-3, 0])

    fig['layout']['xaxis1'].update(title='Maturity (quarters)')
    fig['layout']['xaxis2'].update(title='Maturity (quarters)')
    fig['layout']['xaxis3'].update(title='Maturity (quarters)')

    return fig
