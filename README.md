# UHPC-HRC-LimitStates

[![Python Versions](https://img.shields.io/pypi/pyversions/parametric-uhpc)](https://pypi.org/project/parametric-uhpc)&nbsp;
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A **parametric, closed-form** Python library for ultra-high-performance concrete (**UHPC**) and hybrid-reinforced concrete (**HRC**) flexural limit-state analysis.  
It can calculate

* **Moment–curvature** envelopes  
* **Load–deflection** responses  
* **Internal-force** distributions  

using fully customisable constitutive models for **tension**, **compression**, and **steel reinforcement**.  
Optionally, you can **override any model with experimental Excel files** (flexure, tension, compression, steel).

---

## 🔨 Installation

### 1 · Stable release from PyPI

```bash
pip install parametric-uhpc==1.0.0            # grabs the latest version
```
---

## 🚀 Quick-start

```
# example_full_custom.py

from parametric_uhpc import run_full_model

# Full custom call
results = run_full_model(

    excel_flex        = "***/UHPC-HRC-limitstates/examples/data/flexure.xlsx",
    excel_tension     = "***/UHPC-HRC-limitstates/examples/data/tension.xlsx",
    excel_compression = "***/UHPC-HRC-limitstates/examples/data/compression.xlsx",
    excel_reinforcement = "***/UHPC-HRC-limitstates/examples/data/reinforcement.xlsx",

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
