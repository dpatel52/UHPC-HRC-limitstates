import numpy as np
import matplotlib.pyplot as plt

def calculate_deflection(mom, cv, M_cr, Cmax, Phi_cr, L, Lp, Lp2, Mmax, S2):
    """
    Calculates deflection using the moment–area method.
    """
    delta_total = np.zeros(len(mom))
    first_delta_total = None
    last_delta_total = 0

    for i in range(len(mom)):
        if mom[i] <= M_cr and cv[i] < Cmax and cv[i] <= Phi_cr:
            delta_total[i] = (0.5 * cv[i] * ((L/2) - (S2/2)) *
                              ((2/3)*((L/2) - (S2/2)))) + (cv[i] * (S2/2) *
                              ((L/2) - (S2/4)))
            last_delta_total = delta_total[i]
        elif mom[i] < M_cr and cv[i] <= Cmax and cv[i] > Phi_cr:
            x_cv_cr = (M_cr / mom[i]) * ((L/2) - (Lp/2))
            aa = x_cv_cr
            bb = (L/2) - (Lp/2)
            cc = L/2
            if aa >= bb:
                aa = bb
            x1 = (2/3) * aa
            x2 = ((bb - aa) / 2) + aa
            x3 = aa + ((2/3) * (bb - aa))
            x4 = bb + ((cc - bb) / 2)
            A1 = 0.5 * aa * Phi_cr
            A2 = Phi_cr * (bb - aa)
            A3 = 0.5 * (bb - aa) * (cv[i] - Phi_cr)
            A4 = (cc - bb) * cv[i]
            delta_total[i] = A1 * x1 + A2 * x2 + A3 * x3 + A4 * x4
            last_delta_total = delta_total[i]
        elif mom[i] > M_cr and cv[i] <= Cmax:
            x_cv_cr = (M_cr / mom[i]) * ((L/2) - (Lp/2))
            aa = x_cv_cr
            bb = (L/2) - (Lp/2)
            cc = L/2
            if aa >= bb:
                aa = bb
            x1 = (2/3) * aa
            x2 = ((bb - aa) / 2) + aa
            x3 = aa + ((2/3) * (bb - aa))
            x4 = bb + ((cc - bb) / 2)
            A1 = 0.5 * aa * Phi_cr
            A2 = Phi_cr * (bb - aa)
            A3 = 0.5 * (bb - aa) * (cv[i] - Phi_cr)
            A4 = (cc - bb) * cv[i]
            delta_total[i] = A1 * x1 + A2 * x2 + A3 * x3 + A4 * x4
            last_delta_total = delta_total[i]
        elif mom[i] < Mmax and cv[i] > Cmax:
            aa = (L/2) - (Lp2/2)
            bb = (L/2) - aa
            x1 = (2/3) * aa
            x2 = aa + (bb / 2)
            if aa >= bb:
                aa = bb
            delta1 = x1 * (0.5 * Phi_cr * aa)
            delta2 = x2 * bb * cv[i]
            delta_total[i] = delta1 + delta2
            if first_delta_total is None:
                first_delta_total = delta_total[i]
            delta_total[i] = (last_delta_total - first_delta_total) + delta1 + delta2
    return delta_total

def calculate_deflection_3PB(mom, cv, M_cr, Cmax, Phi_cr, L, Lp, Lp2, Mmax, S2):
    """
    Calculates deflection for 3-point bending.
    """
    delta_total = np.zeros(len(mom))
    first_delta_total = None
    last_delta_total = 0

    if Cmax <= Phi_cr:
        result = cv[cv >= Phi_cr]
        num_points = len(result)
        Lp_fact = np.linspace(1, 1, num_points)  # Array of ones.
        Lp_vect = Lp_fact * Lp
    else:
        result0 = cv[cv <= Phi_cr]
        result = cv[(cv >= Phi_cr) & (cv <= Cmax)]
        result1 = cv[cv >= Cmax]
        num_points0 = len(result0)
        num_points = len(result)
        num_points1 = len(result1)
        Lp_fact0 = np.linspace(0, 0, num_points0)
        Lp_fact = np.linspace(0, 1, num_points)
        Lp_fact1 = np.linspace(1, 1, num_points1)
        Lp_vect = np.concatenate([Lp_fact0 * Lp, Lp_fact * Lp, Lp_fact1 * Lp])
    
    for i in range(len(mom)):
        if mom[i] <= M_cr and cv[i] < Cmax and cv[i] <= Phi_cr:
            delta_total[i] = 0.5 * cv[i] * (L/2) * ((2/3) * (L/2))
            last_delta_total = delta_total[i]
        elif mom[i] < M_cr and cv[i] <= Cmax and cv[i] > Phi_cr:
            x_cv_cr = (M_cr / mom[i]) * ((L/2) - (Lp_vect[i]/2))
            aa = x_cv_cr
            bb = (L/2) - (Lp_vect[i]/2)
            cc = L/2
            if aa >= bb:
                aa = bb
            x1 = (2/3) * aa
            x2 = ((bb - aa) / 2) + aa
            x3 = aa + ((2/3) * (bb - aa))
            x4 = bb + ((cc - bb) / 2)
            A1 = 0.5 * aa * Phi_cr
            A2 = Phi_cr * (bb - aa)
            A3 = 0.5 * (bb - aa) * (cv[i] - Phi_cr)
            A4 = (cc - bb) * cv[i]
            delta_total[i] = A1 * x1 + A2 * x2 + A3 * x3 + A4 * x4
            last_delta_total = delta_total[i]
        elif mom[i] > M_cr and cv[i] <= Cmax:
            x_cv_cr = (M_cr / mom[i]) * ((L/2) - (Lp_vect[i]/2))
            aa = x_cv_cr
            bb = (L/2) - (Lp_vect[i]/2)
            cc = L/2
            if aa >= bb:
                aa = bb
            x1 = (2/3) * aa
            x2 = ((bb - aa) / 2) + aa
            x3 = aa + ((2/3) * (bb - aa))
            x4 = bb + ((cc - bb) / 2)
            A1 = 0.5 * aa * Phi_cr
            A2 = Phi_cr * (bb - aa)
            A3 = 0.5 * (bb - aa) * (cv[i] - Phi_cr)
            A4 = (cc - bb) * cv[i]
            delta_total[i] = A1 * x1 + A2 * x2 + A3 * x3 + A4 * x4
            last_delta_total = delta_total[i]
        elif mom[i] < Mmax and cv[i] > Cmax:
            aa = (L/2) - (Lp2/2)
            bb = (L/2) - aa
            x1 = (2/3) * aa
            x2 = aa + (bb / 2)
            if aa >= bb:
                aa = bb
            delta1 = x1 * (0.5 * Phi_cr * aa)
            delta2 = x2 * bb * cv[i]
            delta_total[i] = delta1 + delta2
            if first_delta_total is None:
                first_delta_total = delta_total[i]
            delta_total[i] = (last_delta_total - first_delta_total) + delta1 + delta2
    return delta_total

def calculate_est(rho, alpha, k_final, beta_all, epsilon_cr, is_bottom):
    """
    Calculates the estimated yield (or ultimate) stress/strain.
    """
    if rho > 0:
        if is_bottom:
            est = (-alpha + k_final) * beta_all * epsilon_cr / (k_final - 1)
        else:
            est = (k_final - 1 + alpha) * beta_all * epsilon_cr / (k_final - 1)
    else:
        est = np.nan
    return est

def find_crossing_rows(est, est_y, est_ult, beta_all, delta_total, label):
    """
    Finds the first index where 'est' crosses the yield and ultimate limits.
    """
    crossing_row1_arr = np.where(est >= est_y)[0]
    crossing_row1 = crossing_row1_arr[0] if crossing_row1_arr.size > 0 else None

    crossing_row2_arr = np.where(est >= est_ult)[0]
    crossing_row2 = crossing_row2_arr[0] if crossing_row2_arr.size > 0 else None

    if crossing_row1 is not None:
        beta_yield = beta_all[crossing_row1]
        defl_yield = delta_total[crossing_row1]
        print(f"{label} yields at beta: {beta_yield}")
    else:
        print(f"No {label} yielding point found")
    if crossing_row2 is not None:
        beta_ult = beta_all[crossing_row2]
        defl_ult = delta_total[crossing_row2]
        print(f"{label} reaches ultimate strain at beta: {beta_ult}")
    else:
        print(f"No {label} ultimate point found")
    return crossing_row1, crossing_row2

def plot_deflection(kappa, epsilon_cr, chi_su, omega, lambda_cu,
                    rho_t, rho_c, beta_all, delta_total, load_4pb,
                    alpha, Envelope, hasFlex, exp_deflection_data, exp_load_data):
    """
    Plots the load–deflection curve and marks key points.
    """
    # In MATLAB, Envelope(:,1) gives the first column.
    k_final = Envelope[:, 0]
    est_y = kappa * epsilon_cr
    est_ult = chi_su * epsilon_cr
    esc_y = omega * epsilon_cr
    esc_ult = lambda_cu * epsilon_cr

    crossing_rows = [None] * 6

    est_bot = calculate_est(rho_t, alpha, k_final, beta_all, epsilon_cr, True)
    est_top = calculate_est(rho_c, alpha, k_final, beta_all, epsilon_cr, False)

    # Check if the calculated estimates are not all NaN.
    if not np.isnan(est_bot).all():
        crossing_rows[0], crossing_rows[1] = find_crossing_rows(est_bot, est_y, est_ult, beta_all, delta_total, 'Bottom steel')
    if not np.isnan(est_top).all():
        crossing_rows[2], crossing_rows[3] = find_crossing_rows(est_top, est_y, est_ult, beta_all, delta_total, 'Top steel')

    ectop = k_final * beta_all * epsilon_cr / (1 - k_final)
    crossing_rows[4], crossing_rows[5] = find_crossing_rows(ectop, esc_y, esc_ult, beta_all, delta_total, 'Concrete in compression')

    plt.figure()
    if hasFlex == 1:
        plt.plot(exp_deflection_data, exp_load_data, '-x',
                 label='Experimental Data', linewidth=1, color='r')
    plt.plot(delta_total, load_4pb, 'b-', label='Simulation', linewidth=1.5)

    markers = ['ro', 'bo', 'go', 'mo', 'ko', 'co']
    labels = ['Bottom Steel Yield', 'Bottom Steel Ultimate',
              'Top Steel Yield', 'Top Steel Ultimate',
              'Concrete Yield', 'Concrete Ultimate']

    for i, crossing in enumerate(crossing_rows):
        if crossing is not None:
            idx = int(crossing)
            plt.plot(delta_total[idx], load_4pb[idx], markers[i],
                     markersize=8, linewidth=2, label=labels[i])
        else:
            plt.plot(np.nan, np.nan, markers[i],
                     markersize=8, linewidth=0.2, label=labels[i] + ' (not found)')

    plt.legend(loc='best')
    plt.xlabel('Deflection (mm)')
    plt.ylabel('Load (N)')
    plt.title('Load-Deflection Plot')
    plt.grid(True)
    plt.show()