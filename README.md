# UHPC-HRC-LimitStates

[![GitHub Release](https://img.shields.io/github/v/release/dpatel52/UHPC-HRC-limitstates)](https://github.com/dpatel52/UHPC-HRC-limitstates/releases)&nbsp;
[![PyPI version](https://badge.fury.io/py/parametric-uhpc.svg)](https://pypi.org/project/parametric-uhpc)&nbsp;
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
pip install parametric-uhpc            # grabs the latest version
```
---

## 🚀 Quick-start

```
from parametric_uhpc import run_full_model

# 1 · Default run – no Excel overrides – plots + returns results
results = run_full_model(plot=True)

# 2 · Custom run – point to your own Excel files *and* tweak geometry/materials
results = run_full_model(
    # Excel overrides (omit or set to None to skip)
    excel_flex="examples/data/flexure.xlsx",
    excel_tension="examples/data/tension.xlsx",
    excel_compression="examples/data/compression.xlsx",
    excel_reinforcement="examples/data/reinforcement.xlsx",

    # ─── Geometry & loading ───────────────────────────────────────
    L=1500.0,      # span (mm)
    b=200.0,       # width
    h=250.0,       # depth
    pointBend=3,   # 3-point (use 4 for four-point)
    S2=500.0,      # load-point separation for 4-PB (ignored for 3-PB)
    Lp=500.0,      # plastic-hinge length (hardening)
    cover=30.0,    # concrete cover

    # ─── Tension model (UHPC) ─────────────────────────────────────
    E=35000.0,             # MPa
    epsilon_cr=1e-4,       # first-crack strain
    mu_1=0.30, mu_2=0.30, mu_3=0.30,
    beta_1=1.01, beta_2=25.0, beta_3=300.0,

    # ─── Compression model (UHPC) ────────────────────────────────
    xi=1.05,   # Ex/E
    omega=8.0,
    mu_c=1.0,
    ecu=0.0035,

    # ─── Steel model ─────────────────────────────────────────────
    Es=210000.0,            # MPa
    kappa=(550e6/210000e6)/1e-4,   # (fy/Es)/εcr  for fy = 550 MPa
    mu_s=1.1,

    # ─── Reinforcement layout ────────────────────────────────────
    botDiameter=12.0, botCount=3,
    topDiameter=10.0, topCount=2,

    plot=True             # turn plots on/off
)
