# Parametric UHPC Flexural Model

[![PyPI version](https://badge.fury.io/py/parametric-uhpc.svg)](https://pypi.org/project/parametric-uhpc)  
[![Python Versions](https://img.shields.io/pypi/pyversions/parametric-uhpc)](https://pypi.org/project/parametric-uhpc)  
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A **parametric**, **closed-form** Python package for ultra-high-performance concrete (UHPC) flexural limit-state analysis.  
Compute:
- **Moment–curvature envelopes**  
- **Load–deflection responses** (3- and 4-point bending)  
- **Internal force distributions**  

All with fully customizable tension, compression, and reinforcement constitutive models.

---

## Installation

```bash
# From PyPI
pip install parametric-uhpc

# Or install latest development version
git clone https://github.com/yourusername/parametric-uhpc.git
cd parametric-uhpc
pip install -e .
