import numpy as np
import warnings

def calculate_envelope_new_2(rho_c,rho_t,kappa, omega, epsilon_cr, beta_all, 
                             k111, M111, 
                             k211, M211, k212, M212, 
                             k221, M221, k222, M222, 
                             k311, M311, k312, M312, 
                             k321, M321, k322, M322, 
                             k411, M411, k412, M412, 
                             k421, M421, k422, M422, 
                             k4222, M4222,
                             beta_1, beta_2, beta_3, alpha, 
                             T, C, R, RC):
    """
    Calculate the Envelope array for a given beta vector and stage parameters.
    The zones T, C, R, RC are updated based on thresholds.
    
    Parameters:
      kappa, omega, epsilon_cr : scalar parameters
      beta_all                 : 1D NumPy array of beta values for this call
      k111, M111, etc.         : 1D NumPy arrays for each stage
      beta_1, beta_2, beta_3   : thresholds for updating zones
      alpha                    : geometric offset
      T, C, R, RC              : current zone indicators (initially 1)
      
    Returns:
      Envelope : 2D NumPy array with shape (n, 2) where each row is [k, M]
      T, C, R, RC : updated zone indicators after processing all increments.
    """
    # Create a persistent variable for warning display.
    if not hasattr(calculate_envelope_new_2, "warning_displayed"):
        calculate_envelope_new_2.warning_displayed = False

    n = beta_all.shape[0]
    Envelope = np.zeros((n, 2))
    econ_y = np.zeros(n)   # top fiber strain (concrete compression zone)
    est_y  = np.zeros(n)   # steel tension strain
    esc_y  = np.zeros(n)   # compression steel strain

    # Loop over each increment in beta_all
    for i in range(n):
        # Construct the stage string from current zone indicators.
        
        if rho_c == 0:
            RC = 1
        if rho_t == 0:
            R = 1

        stageStr = f"{T}{C}{R}{RC}"

        # Select the proper stage arrays based on stageStr.
        if stageStr == '1111':
            kStage = k111
            MStage = M111
        elif stageStr == '2111':
            kStage = k211
            MStage = M211
        elif stageStr == '2121':
            kStage = k212
            MStage = M212
        elif stageStr == '2211':
            kStage = k221
            MStage = M221
        elif stageStr == '2221':
            kStage = k222
            MStage = M222
        elif stageStr == '3111':
            kStage = k311
            MStage = M311
        elif stageStr == '3121':
            kStage = k312
            MStage = M312
        elif stageStr == '3211':
            kStage = k321
            MStage = M321
        elif stageStr == '3221':
            kStage = k322
            MStage = M322
        elif stageStr == '4111':
            kStage = k411
            MStage = M411
        elif stageStr == '4121':
            kStage = k412
            MStage = M412
        elif stageStr == '4211':
            kStage = k421
            MStage = M421
        elif stageStr == '4221':
            kStage = k422
            MStage = M422
        elif stageStr == '4222':
            kStage = k4222
            MStage = M4222
        else:
            warnings.warn(f'Stage "{stageStr}" not recognized')
            # Optionally, you might choose to continue or break
            continue

        # Get the current k and M for the i-th increment.
        # (Assuming the stage arrays have at least n elements.)
        k_i = kStage[i]
        M_i = MStage[i]
        Envelope[i, 0] = k_i   # k(i)
        Envelope[i, 1] = M_i   # M(i)

        beta_i = beta_all[i]

        # Strain calculations
        # Note: In the formulas below, we assume k_i != 1.
        econ_y[i] = abs(k_i * beta_i * epsilon_cr / (1 - k_i))
        est_y[i]  = abs(((-alpha + k_i) * beta_i * epsilon_cr) / (k_i - 1))
        esc_y[i]  = abs(((k_i - 1 + alpha) * beta_i * epsilon_cr) / (k_i - 1))

        # Update the tension zone T
        if T == 1 and beta_i >= 1:
            T = 2
        elif T == 2 and beta_i >= beta_1:
            T = 3
        elif T == 3 and beta_i >= beta_2:
            T = 4

        # Update the compression zone C (only from 1 to 2)
        if C == 1 and econ_y[i] >= (omega * epsilon_cr):
            C = 2

        # Update the steel zone R (from 1 to 2 if yield)
        if R == 1 and est_y[i] >= kappa * epsilon_cr:
            R = 2

        # Update RC for compression steel
        # Note the MATLAB logic:
        # if (RC == 1) && (esc_y(i) >= kappa * epsilon_cr) && strcmp(stageStr, '4221') || strcmp(stageStr,'4222')
        #     RC = 2;
        # end
        if RC == 1 and esc_y[i] >= kappa * epsilon_cr and (stageStr == '4221' or stageStr == '4222'):
            RC = 2

        if RC == 1 and esc_y[i] >= kappa * epsilon_cr and not calculate_envelope_new_2.warning_displayed:
            warnings.warn('Compression Steel assumed Elastic because it yields while tension steel or concrete in compression is still elastic. Reduce Tension Steel.')
            calculate_envelope_new_2.warning_displayed = True

    return Envelope, T, C, R, RC
