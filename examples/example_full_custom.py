# example_full_custom.py

from parametric_uhpc import run_full_model

# Full custom call
results = run_full_model(

    excel_flex        = "I:/My Drive/PhD + Research/PhD/03 Code Conversion/UHPC-HRC-limitstates/examples/data/flexure.xlsx",
    excel_tension     = "I:/My Drive/PhD + Research/PhD/03 Code Conversion/UHPC-HRC-limitstates/examples/data/tension.xlsx",
    excel_compression = "I:/My Drive/PhD + Research/PhD/03 Code Conversion/UHPC-HRC-limitstates/examples/data/compression.xlsx",
    excel_reinforcement = "I:/My Drive/PhD + Research/PhD/03 Code Conversion/UHPC-HRC-limitstates/examples/data/reinforcement.xlsx",

    # Geometry & loadingY
    L         = 1092.0,    # mm
    b         = 101.0,     # mm
    h         = 203.0,     # mm
    pointBend = 4,         # 3 or 4 point bend only
    S2        = 254.0,     # mm distance between load points
    Lp        = 254.0,     # mm plastic length (if unknown use Lp=S2 for 4PB and Lp=d for 3PB)
    cLp        = 125.0,     # mm post-localization plastic length (use cLp = d if unknown)
    cover     = 38.0,      # mm concrete cover

    # Material (tension)
    E = 45526.0,
    epsilon_cr = 0.00015,
    sigma_t1 = 11.5,
    sigma_t2 = 5.5,
    sigma_t3 = 1.5,
    epsilon_t1 = 0.003,
    epsilon_t2 = 0.02,
    epsilon_t3 = 0.08,

    # Material (compression)
    Ec = 45526.0 * 1.01,
    sigma_cy = 200,
    sigma_cu = 30,
    ecu = 0.025,

    # Steel
    Es = 200000.0,
    fsy = 460,
    fsu = 670,
    epsilon_su = 0.14,

    # Reinforcement geometry
    botDiameter    = (3/8)*25.4, # mm
    botCount       = 2,
    topDiameter    = 10.0,
    topCount       = 0,

    # Turn off plotting if you just want numbers
    plot      = True
)

# Summarize the results
print("→ Peak moment (N·mm):", results["moment"].max())
print("→ Peak Load (N):", results["load"].max())
