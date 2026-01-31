# MCM 2026 Problem A: Smartphone Battery Discharge Model

## Overview

This repository contains a comprehensive solution for the 2026 Mathematical Contest in Modeling (MCM) Problem A: **Smartphone Battery Discharge Modeling**.

We develop a physics-based continuous-time mathematical model for lithium-ion battery discharge in smartphones under realistic usage conditions.

## Project Structure

```
mcm_solution/
├── src/
│   ├── battery_model.py      # Core battery discharge model
│   ├── sensitivity_analysis.py # Sensitivity and uncertainty analysis
│   ├── visualization.py       # Figure generation
│   └── main.py               # Main analysis script
├── docs/
│   └── mcm_solution.tex      # Complete LaTeX solution document
├── figures/                   # Generated figures
│   ├── discharge_curves.png
│   ├── power_breakdown.png
│   ├── temperature_effects.png
│   ├── voltage_soc.png
│   ├── aging_effects.png
│   ├── validation.png
│   ├── sensitivity_tornado.png
│   ├── monte_carlo.png
│   └── recommendations.png
├── data/                      # Output data files
├── requirements.txt           # Python dependencies
└── README.md
```

## Mathematical Model

### Core Governing Equation

The State of Charge (SOC) evolves according to:

$$\frac{d\text{SOC}}{dt} = -\frac{P(t)}{V(\text{SOC}, T) \cdot Q_{\text{eff}}(T, \text{SOH})}$$

Where:
- $P(t)$ is total power consumption (W)
- $V$ is terminal voltage (V)
- $Q_{\text{eff}}$ is effective capacity (mAh)
- $T$ is temperature (°C)
- SOH is State of Health (battery aging factor)

### Power Consumption Model

Total power is the sum of components:

$$P(t) = P_{\text{screen}} + P_{\text{CPU}} + P_{\text{GPU}} + P_{\text{network}} + P_{\text{baseline}}$$

### Temperature Effects

Uses Arrhenius-type temperature dependence for:
- Effective capacity (decreases at low temperatures)
- Internal resistance (increases at low temperatures)

### Battery Aging

Capacity fade from calendar and cycle aging:

$$\text{SOH} = 1 - L_{\text{cal}}(t) - L_{\text{cyc}}(n)$$

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete analysis:

```bash
cd src
python3 main.py
```

This will:
1. Simulate battery discharge for multiple usage scenarios
2. Perform sensitivity analysis
3. Generate all figures for the report
4. Print summary results and recommendations

### Using the Model in Python

```python
from battery_model import SmartphoneBatteryModel, UsageProfile

# Create model with default parameters
model = SmartphoneBatteryModel()

# Define a usage profile
profile = UsageProfile(
    screen_brightness=lambda t: 0.5,  # 50% brightness
    screen_on=lambda t: 1.0,          # Screen always on
    cpu_load=lambda t: 0.3,           # 30% CPU load
    temperature=lambda t: 25.0        # 25°C
)

# Simulate discharge from 100% SOC
t, SOC, details = model.simulate(1.0, profile, t_max=24.0)

print(f"Time to empty: {details['t_empty']:.2f} hours")
```

## Key Results

### Usage Scenario Comparison

| Scenario | Avg. Power (W) | Time-to-Empty (h) |
|----------|----------------|-------------------|
| Idle | 0.52 | 32.0 |
| Light | 1.75 | 9.6 |
| Moderate | 4.35 | 4.1 |
| Video Call | 5.53 | 3.6 |
| Navigation | 5.74 | 3.3 |
| Heavy Gaming | 10.12 | 1.9 |

### Sensitivity Analysis

Most impactful parameters:
1. **Battery capacity** (elasticity: +0.99)
2. **Temperature** (elasticity: +0.64)
3. **Screen brightness** (elasticity: -0.32)
4. **CPU load** (elasticity: -0.31)

### Recommendations

1. **Reduce screen brightness** → +29% battery life
2. **Close background apps** → +36% battery life
3. **Use WiFi over cellular** → Modest improvement
4. **All optimizations combined** → +115% battery life

## Building the Report

Compile the LaTeX document:

```bash
cd docs
pdflatex mcm_solution.tex
```

## References

1. Tremblay, O., Dessaint, L.-A. (2009). Experimental validation of a battery dynamic model for EV applications.
2. Xu, B., et al. (2016). Modeling of lithium-ion battery degradation for cell life assessment.
3. Carroll, A., Heiser, G. (2010). An analysis of power consumption in a smartphone.

## License

This code is provided for educational purposes as part of the MCM competition solution.
