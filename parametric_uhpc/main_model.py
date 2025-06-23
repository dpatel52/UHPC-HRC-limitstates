import numpy as np
import matplotlib.pyplot as plt

from .mk_equations import (
    stage111, stage211, stage212, stage221, stage222,
    stage311, stage312, stage321, stage322,
    stage411, stage412, stage421, stage422, stage4222
)
from .envelope import calculate_envelope_new_2
from .deflection import calculate_deflection, calculate_deflection_3PB, plot_deflection
from .draw import draw_doubly_reinforced_beam
from .plot_beta import plotBetaVsMandK


def run_full_model(
    *,
    # Experimental-data Excel paths
    excel_flex: str = None,
    excel_tension: str = None,
    excel_compression: str = None,
    excel_reinforcement: str = None,
    # Geometry & loading
    L: float = 1000.0,
    b: float = 120.0,
    h: float = 175.0,
    pointBend: int = 3,
    S2: float = 125.0,
    Lp: float = 125.0,
    cLp: float = 125.0,
    cover: float = 25.0,
    # Material (tension)
    E: float = 34000.0,
    epsilon_cr: float = 0.00012,
    sigma_t1: float = 10,
    sigma_t2: float = 6,
    sigma_t3: float = 2,
    epsilon_t1: float = 0.00012*3.0,
    epsilon_t2: float = 0.00012*30.0,
    epsilon_t3: float = 0.00012*290.0,
    # Material (compression)
    Ec: float = 34000.0*1.01,
    sigma_cy: float = 50,
    sigma_cu: float = 50,
    ecu: float = 0.003,
    # Steel
    Es: float = 200000.0,
    fsy: float = 60,
    fsu: float = 65.0,
    epsilon_su: float = 0.09,
    # Reinforcement geometry
    botDiameter: float = 12.0,
    botCount: int = 2,
    topDiameter: float = 10.0,
    topCount: int = 2,
    # Plot flag
    plot: bool = True,
) -> dict:
    
    """
    Run the full UHPC flexural-limit-states simulation.
    Optional Excel inputs for experimental data and model overrides.
    Returns a dict containing arrays and key results.
    """

    # ------------------------------------------------------------------------
    #convert to model parameters
    sigma_cr = E * epsilon_cr
    mu_1 = sigma_t1 / sigma_cr
    mu_2 = sigma_t2 / sigma_cr
    mu_3 = sigma_t3 / sigma_cr
    beta_1 = epsilon_t1 / epsilon_cr
    beta_2 = epsilon_t2 / epsilon_cr
    beta_3 = epsilon_t3 / epsilon_cr

    xi = Ec / E
    omega = sigma_cy / (Ec * epsilon_cr)
    mu_c = sigma_cu / sigma_cy
    
    
    
    kappa = (fsy / Es) / epsilon_cr
    mu_s = fsu / fsy
    chi_su = epsilon_su / epsilon_cr


    # --- Experimental data loading ---
    exp_load_data = []
    exp_deflection_data = []
    import pandas as pd
    if excel_flex:
        df_f = pd.read_excel(excel_flex)
        exp_deflection_data = np.array(df_f.iloc[:,0])
        exp_load_data = np.array(df_f.iloc[:,1])
    
    # --- Custom material model override via Excel ---
    tension = None
    if excel_tension:
        dt = pd.read_excel(excel_tension)
        tension = {
            'strain': dt.iloc[:,0].to_numpy(),
            'stress': dt.iloc[:,1].to_numpy()
        }
    compression = None
    if excel_compression:
        dc = pd.read_excel(excel_compression)
        compression = {
            'strain': dc.iloc[:,0].to_numpy(),
            'stress': dc.iloc[:,1].to_numpy()
        }
    reinforcement = None
    if excel_reinforcement:
        dr = pd.read_excel(excel_reinforcement)
        reinforcement = {
            'strain': dr.iloc[:,0].to_numpy(),
            'stress': dr.iloc[:,1].to_numpy()
        }

    # --- Derived geometry & parameters ---
    alpha = (h - cover) / h


    # --- Slopes ---
    eta_1 = (mu_1 - 1) / (beta_1 - 1)
    eta_2 = (mu_2 - mu_1) / (beta_2 - beta_1)
    eta_3 = (mu_3 - mu_2) / (beta_3 - beta_2)
    lambda_cu = ecu / epsilon_cr
    eta_c = (mu_c - 1) / (lambda_cu - omega)
    n = Es / E
    eta_s = (mu_s - 1) / (chi_su - kappa)
    
    # --- Reinforcement ratios ---
    s_area_bot = botCount * (botDiameter**2 * np.pi / 4)
    rho_t = s_area_bot / (b * h)
    s_area_top = topCount * (topDiameter**2 * np.pi / 4)
    rho_c = s_area_top / (b * h)

    # ------------------------------------------------------------------------
    # Plot or use custom models
    # ------------------------------------------------------------------------
    if plot:
        # Tension
        if tension:
            strainT_exp, stressT_exp = tension['strain'], tension['stress']
        else:
            strainT_exp, stressT_exp = None, None
        strainT = [0, epsilon_cr, epsilon_cr*beta_1, epsilon_cr*beta_2, epsilon_cr*beta_3]
        stressT = [0, epsilon_cr*E, mu_1*epsilon_cr*E, mu_2*epsilon_cr*E, mu_3*epsilon_cr*E]
        # Compression
        if compression:
            strainC_exp, stressC_exp = compression['strain'], compression['stress']
        else:
            strainC_exp, stressC_exp = None, None
        strainC = [0, omega*epsilon_cr, lambda_cu*epsilon_cr]
        stressC = [0, E*epsilon_cr*xi*omega, E*epsilon_cr*xi*omega*mu_c]
        # Reinforcement
        if reinforcement:
            strainR_exp, stressR_exp = reinforcement['strain'], reinforcement['stress']
        else:
            strainR_exp, stressR_exp = None, None
        strainR = [0, kappa*epsilon_cr, chi_su*epsilon_cr]
        stressR = [0, E*n*kappa*epsilon_cr, E*n*kappa*epsilon_cr*mu_s]

        fig, axs = plt.subplots(1,3,figsize=(18,5))
        fig.suptitle("Input Models")

        # Tension plot
        axs[0].plot(strainT, stressT, '-o', color='red', label='Calibrated Tension (Matrix)')
        if strainT_exp is not None and stressT_exp is not None:
            axs[0].plot(strainT_exp, stressT_exp, 'x--', color='black', label='Experimental Tension')
        axs[0].set_xlabel("Strain (mm/mm)")
        axs[0].set_ylabel("Stress (MPa)")
        axs[0].grid(True)
        axs[0].legend()

        # Compression plot
        axs[1].plot(strainC, stressC, '-o', color='blue', label='Calibrated Compression (Matrix)')
        if strainC_exp is not None and stressC_exp is not None:
            axs[1].plot(strainC_exp, stressC_exp, 'x--', color='black', label='Experimental Compression')
        axs[1].set_xlabel("Strain (mm/mm)")
        axs[1].set_ylabel("Stress (MPa)")
        axs[1].grid(True)
        axs[1].legend()

        # Reinforcement plot
        axs[2].plot(strainR, stressR, '-o', color='green', label='Calibrated Steel (Reinforcement)')
        if strainR_exp is not None and stressR_exp is not None:
            axs[2].plot(strainR_exp, stressR_exp, 'x--', color='black', label='Experimental Steel')
        axs[2].set_xlabel("Strain (mm/mm)")
        axs[2].set_ylabel("Stress (MPa)")
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------------
    # Cracking properties
    # ------------------------------------------------------------------------
    kcr, _ = stage111(
        1, L, b, h, alpha, E, epsilon_cr,
        beta_1, beta_2, beta_3, eta_1, eta_2, eta_3,
        xi, omega, eta_c, n, kappa, eta_s, rho_c, rho_t
    )
    M_cr = epsilon_cr*E*b*h**2 / (12*(1 - kcr))
    Phi_cr = epsilon_cr / ((1 - kcr)*h)

    # ------------------------------------------------------------------------
    # Prepare beta vectors and stage envelopes
    # ------------------------------------------------------------------------
    beta_z1 = np.linspace(0,1,1000)
    beta_z2 = np.linspace(1,beta_1,2000)
    beta_z3 = np.linspace(beta_1,beta_2,2000)
    beta_z4 = np.linspace(beta_2,beta_3,2000)

    k111, M111 = stage111(beta_z1, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k211, M211 = stage211(beta_z2, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k212, M212 = stage212(beta_z2, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k221, M221 = stage221(beta_z2, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k222, M222 = stage222(beta_z2, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k311, M311 = stage311(beta_z3, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k312, M312 = stage312(beta_z3, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k321, M321 = stage321(beta_z3, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k322, M322 = stage322(beta_z3, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k411, M411 = stage411(beta_z4, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k412, M412 = stage412(beta_z4, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k421, M421 = stage421(beta_z4, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k422, M422 = stage422(beta_z4, L, b, h, alpha, E, epsilon_cr,
                          beta_1, beta_2, beta_3,
                          eta_1, eta_2, eta_3,
                          xi, omega, eta_c,
                          n, kappa, eta_s,
                          rho_c, rho_t)
    k4222, M4222 = stage4222(beta_z4, L, b, h, alpha, E, epsilon_cr,
                            beta_1, beta_2, beta_3,
                            eta_1, eta_2, eta_3,
                            xi, omega, eta_c,
                            n, kappa, eta_s,
                            rho_c, rho_t)

    # build envelope
    T = C = R = RC = 1
    Envelope1, T, C, R, RC = calculate_envelope_new_2(
        rho_c, rho_t, kappa, omega, epsilon_cr, beta_z1,
        k111, M111, k211, M211, k212, M212,
        k221, M221, k222, M222,
        k311, M311, k312, M312,
        k321, M321, k322, M322,
        k411, M411, k412, M412,
        k421, M421, k422, M422,
        k4222, M4222,
        beta_1, beta_2, beta_3, alpha,
        T, C, R, RC
    )
    Envelope2, T, C, R, RC = calculate_envelope_new_2(
        rho_c, rho_t, kappa, omega, epsilon_cr, beta_z2,
        k111, M111, k211, M211, k212, M212,
        k221, M221, k222, M222,
        k311, M311, k312, M312,
        k321, M321, k322, M322,
        k411, M411, k412, M412,
        k421, M421, k422, M422,
        k4222, M4222,
        beta_1, beta_2, beta_3, alpha,
        T, C, R, RC
    )
    Envelope3, T, C, R, RC = calculate_envelope_new_2(
        rho_c, rho_t, kappa, omega, epsilon_cr, beta_z3,
        k111, M111, k211, M211, k212, M212,
        k221, M221, k222, M222,
        k311, M311, k312, M312,
        k321, M321, k322, M322,
        k411, M411, k412, M412,
        k421, M421, k422, M422,
        k4222, M4222,
        beta_1, beta_2, beta_3, alpha,
        T, C, R, RC
    )
    Envelope4, T, C, R, RC = calculate_envelope_new_2(
        rho_c, rho_t, kappa, omega, epsilon_cr, beta_z4,
        k111, M111, k211, M211, k212, M212,
        k221, M221, k222, M222,
        k311, M311, k312, M312,
        k321, M321, k322, M322,
        k411, M411, k412, M412,
        k421, M421, k422, M422,
        k4222, M4222,
        beta_1, beta_2, beta_3, alpha,
        T, C, R, RC
    )

    Envelope = np.vstack((Envelope1, Envelope2, Envelope3, Envelope4))
    beta_all = np.concatenate((beta_z1, beta_z2, beta_z3, beta_z4))

    # compute curvature & moment
    strain_bot = beta_all * epsilon_cr
    NA_from_bot = (1 - Envelope[:,0]) * h
    Curvature = strain_bot / NA_from_bot
    Moment    = Envelope[:,1]

     # ------------------------------------------------------------------------
    # Moment-Curvature plot
    # ------------------------------------------------------------------------
    if plot:
        # Calculate limit state strains and curvatures
        k_final = Envelope[:, 0]
        est_bot = (-alpha + k_final) * beta_all * epsilon_cr / (k_final - 1)
        est_top = np.abs((k_final - 1 + alpha) * beta_all * epsilon_cr / (k_final - 1))
        strain_top = k_final * beta_all * epsilon_cr / (1 - k_final)

        # Find limit state curvatures
        def find_first_idx(arr, val):
            idx = np.where(arr >= val)[0]
            return idx[0] if idx.size > 0 else None

        service_curv_idx = find_first_idx(est_bot, 0.8 * kappa * epsilon_cr)
        bot_st_yld_curv_idx = find_first_idx(est_bot, kappa * epsilon_cr)
        bot_st_ult_curv_idx = find_first_idx(est_bot, chi_su * epsilon_cr)
        top_st_yld_curv_idx = find_first_idx(est_top, kappa * epsilon_cr)
        top_st_ult_curv_idx = find_first_idx(est_top, chi_su * epsilon_cr)
        comp_yld_curv_idx = find_first_idx(strain_top, omega * epsilon_cr)
        comp_ult_curv_idx = find_first_idx(strain_top, lambda_cu * epsilon_cr)
        crack_curv = Phi_cr
        crack_local_curv_idx = np.argmax(Envelope[:, 1])  # max moment

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(Curvature, Moment, '-r', label="Simulation", linewidth=2)

        # Markers/colors
        markers = [
            ('o', 'red', 'First Crack', crack_curv, None),
            ('o', 'black', 'Service Stress', None, service_curv_idx),
            ('o', 'blue', 'Steel Yield', None, bot_st_yld_curv_idx),
            ('o', 'green', 'Steel Ultimate Strain', None, bot_st_ult_curv_idx),
            ('o', 'magenta', 'Top Steel Yield', None, top_st_yld_curv_idx if rho_c > 0 else None),
            ('o', 'cyan', 'Top Steel Ultimate', None, top_st_ult_curv_idx),
            ('s', 'brown', 'Crack Localization', None, crack_local_curv_idx),
            ('s', 'purple', 'Conc. Compression Yield', None, comp_yld_curv_idx),
            ('s', 'orange', 'Conc. Compression Ultimate', None, comp_ult_curv_idx),
        ]

        # Plot markers
        for marker, color, label, fixed_curv, idx in markers:
            if fixed_curv is not None:
                # For first crack, use Phi_cr
                y_val = np.interp(fixed_curv, Curvature, Moment)
                plt.plot(fixed_curv, y_val, marker=marker, color=color, markersize=12, label=label, linewidth=2)
            elif idx is not None:
                plt.plot(Curvature[idx], Moment[idx], marker=marker, color=color, markersize=12, label=label, linewidth=2)

        plt.xlabel("Curvature (1/mm)", fontsize=14, fontweight='bold')
        plt.ylabel("Bending Moment (N-mm)", fontsize=14, fontweight='bold')
        plt.title("Moment vs Curvature")
        plt.legend(loc='best', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        mu = (mu_1 + mu_2 + mu_3)/3
        draw_doubly_reinforced_beam(L,h, b, cover,
                                   topDiameter, topCount,
                                   botDiameter, botCount,
                                   mu,sideview=True, loadingType=pointBend, loadSpacing=S2)

    # ------------------------------------------------------------------------
    # Load & deflection (with flex data)
    # ------------------------------------------------------------------------
    if pointBend==3:
        load_4pb = 4*Moment / L
    else:
        load_4pb = 2*Moment / ((L - S2)/2)

    idx = np.argmax(Moment)
    Cmax = Curvature[idx]

    if pointBend==3:
        delta_total = calculate_deflection_3PB(
            Moment, Curvature, M_cr, Cmax, Phi_cr,
            L, Lp, cLp, Moment[idx], S2)
    else:
        delta_total = calculate_deflection(
            Moment, Curvature, M_cr, Cmax, Phi_cr,
            L, Lp, cLp, Moment[idx], S2)

    if plot:
        # only plot experimental if we actually loaded some points
        has_exp = exp_load_data is not None and len(exp_load_data) > 0
        plot_deflection(
            kappa, epsilon_cr, chi_su,
            omega, lambda_cu,
            rho_t, rho_c,
            beta_all, delta_total,
            load_4pb, alpha,
            Envelope, has_exp,
            exp_deflection_data, exp_load_data
        )


    return {
        "beta": beta_all,
        "moment": Moment,
        "curvature": Curvature,
        "load": load_4pb,
        "deflection": delta_total,
        "M_cr": M_cr,
        "Phi_cr": Phi_cr,
    }
