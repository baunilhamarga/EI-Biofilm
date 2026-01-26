# EI-Biofilm
This is the code for the EI "Biofouling an obstacle to electricity production coupled with an environmental risk"

# 1SC2292-ST2-EI — Biofouling & Heat Exchanger Efficiency Model

This repository contains a Python project developed in the context of **1SC2292-ST2-EI** (ST2 Thematic Sequence: *Bioengineering — Produce, Protect, Repair*).

## Project overview

Nuclear power plants use three cooling circuits in series. The **tertiary circuit** withdraws water from the environment (river or sea) to remove heat from the secondary circuit by condensing steam. Because this water is discharged back to the environment, its **thermal, physicochemical, and biological** alterations must remain limited.

However, natural microorganisms present in intake water can grow inside condenser tubes and form **biofilms** (biofouling). This creates four linked issues:

- **Operational / financial:** biofilms reduce heat transfer, lowering efficiency and increasing cooling needs (higher cost).
- **Environmental (water resource):** discharge temperature must remain acceptable for aquatic ecosystems; meeting this constraint can require more water withdrawal, which conflicts with low river flows during droughts.
- **Environmental (chemical pollution):** chemical treatments can reduce fouling but generate pollutant discharge.
- **Health:** without treatment, microorganisms (including potential pathogens) may proliferate and be released at higher concentrations near the outlet.

The course objective is to **model the loss of heat exchanger efficiency caused by biofouling**, then use simulation results to propose **operating and investment scenarios** that balance performance, environmental constraints, and health considerations. **KPIs** are used to compare scenarios and quantify value created.

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt` (install with `pip install -r requirements.txt`)

## How to run

From the repository root:

```bash
python main.py
```

## Course context

**Course:** Biofouling: an obstacle to electricity production coupled with an environmental risk
**Credits:** 2.5 ECTS
**Format:** 5-day problem-solving course (modeling → simulation → validation → industrial case study)

