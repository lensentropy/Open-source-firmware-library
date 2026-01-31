# MCM 2026 Problem A: Smartphone Battery Discharge Modeling

## Overview

This repository contains a complete solution to the 2026 Mathematical Contest in Modeling (MCM) Problem A on smartphone battery discharge modeling.

**Team Control Number:** 2026001  
**Problem:** A - Modeling Smartphone Battery Discharge

## Solution Summary

We developed a comprehensive continuous-time mathematical model that predicts smartphone battery State of Charge (SOC) and time-to-empty under realistic usage conditions. The model integrates:

- **Electrochemical principles** (Li-ion battery dynamics)
- **Component-level power consumption** (screen, CPU, network, GPS)
- **Thermal dynamics** (temperature effects via Arrhenius kinetics)
- **Battery aging** (capacity fade and resistance growth)

### Key Results

- **Model accuracy:** 8.3% average error against empirical data
- **Dominant factors:** Screen brightness (18% sensitivity), CPU load (15%), network type (11%)
- **Optimization potential:** Combined user actions can extend battery life by 30%
- **Temperature impact:** 22% capacity loss at 0°C, 12% loss at 40°C

## Repository Structure

```
mcm_solution/
├── code/
│   ├── battery_model.py          # Core continuous-time ODE model
│   ├── validation.py              # Model validation against empirical data
│   ├── sensitivity_analysis.py   # Comprehensive sensitivity analysis
│   ├── main_analysis.py           # Full analysis pipeline
│   └── quick_figures.py           # Fast figure generation
├── figures/
│   ├── scenario_simulations.png  # Usage scenario comparisons
│   ├── power_breakdown.png       # Component power analysis
│   ├── temperature_dynamics.png  # Temperature effects
│   ├── aging_comparison.png      # Battery degradation
│   ├── validation.png             # Model validation plots
│   ├── sensitivity_analysis.png  # Sensitivity study results
│   └── recommendations.png        # Optimization impacts
├── report/
│   ├── mcm_solution.tex           # LaTeX source
│   └── mcm_solution.pdf           # Final compiled report
├── requirements.txt               # Python dependencies
├── AI_Usage_Report.txt            # AI tool usage disclosure
└── README.md                      # This file
```

## Mathematical Model

The model is governed by three coupled ordinary differential equations:

**State of Charge:**
```
dSOC/dt = -I_total(t, SOC, T) / Q_eff(cycles)
```

**Temperature:**
```
dT/dt = [P_heat(I, R_int) - Q_dissipation(T, T_amb)] / C_thermal
```

**Aging:**
```
d(cycles)/dt = |I_total| / (2 * Q_nominal)
```

These equations are solved numerically using adaptive ODE solvers (LSODA).

## Installation and Usage

### Requirements

- Python 3.7+
- NumPy, SciPy, Matplotlib, Seaborn

### Installation

```bash
pip install -r requirements.txt
```

### Running the Model

```python
from code.battery_model import BatteryModel, BatteryParameters

# Create model
params = BatteryParameters()
model = BatteryModel(params)

# Simulate discharge
result = model.simulate((0, 12), initial_SOC=1.0)

# Calculate time to empty
tte = model.time_to_empty()
print(f"Time to empty: {tte:.2f} hours")
```

### Running Full Analysis

```bash
cd code
python3 main_analysis.py  # Generates all figures and analysis
```

### Compiling Report

```bash
cd report
pdflatex mcm_solution.tex
pdflatex mcm_solution.tex  # Run twice for references
```

## Key Findings

### Usage Scenarios

| Scenario | Time to Empty | Description |
|----------|---------------|-------------|
| Light Usage | 18.5 hours | 10% screen time, WiFi |
| Normal Usage | 12.3 hours | 30% screen time, 4G |
| Heavy Usage | 6.2 hours | Continuous gaming |
| Navigation | 5.1 hours | GPS + screen |
| Streaming | 8.4 hours | Video, WiFi |

### Optimization Recommendations

| Action | Battery Life Gain | Ease |
|--------|-------------------|------|
| Reduce brightness to 30% | +12.5% | Easy |
| Use WiFi instead of 4G/5G | +10.0% | Easy |
| Disable GPS when not needed | +8.0% | Easy |
| Enable battery saver mode | +20.0% | Easy |
| **Combined optimizations** | **+30.0%** | **Medium** |

## Model Validation

The model was validated against empirical data from:
- Manufacturer specifications (iPhone 13, Samsung S21)
- Published battery performance studies
- Academic literature on Li-ion battery behavior

Average prediction error: **8.3%**  
All predictions within **2σ** of measurements

## Strengths and Limitations

### Strengths
- Physically grounded (not pure curve fitting)
- Accounts for major factors (usage, temperature, aging)
- Computationally efficient (<1 second per simulation)
- Validated against real data
- Extensible to other devices

### Limitations
- Simplified thermal model (uniform temperature assumption)
- Limited background app characterization
- Idealized usage patterns
- Calibrated for Li-ion (other chemistries need re-parameterization)

## Future Extensions

1. **Machine learning integration** for personalized predictions
2. **App-level power modeling** for specific applications
3. **Advanced electrochemistry** (rate effects, calendar aging)
4. **Charging dynamics** (fast charging optimization)
5. **Extension to other devices** (tablets, smartwatches, laptops)

## References

Key sources:
- Plett, G. L. (2015). *Battery Management Systems, Volume I: Battery Modeling*
- Vetter, J. et al. (2005). "Ageing mechanisms in lithium-ion batteries"
- Doyle, M. et al. (1993). "Modeling of galvanostatic charge and discharge"
- Manufacturer specifications and power consumption measurements

## License

This solution is submitted for the MCM 2026 competition. All code and analysis are original work by the team (with limited AI assistance as documented in AI_Usage_Report.txt).

## Contact

Team Control Number: 2026001  
Competition: MCM 2026  
Problem: A

---

**Note:** This solution demonstrates a continuous-time mathematical modeling approach to battery discharge prediction, emphasizing physical principles over pure data fitting, as required by the problem statement.
