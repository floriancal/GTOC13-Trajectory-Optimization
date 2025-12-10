# GTOC13-Trajectory-Optimization

This repository contains trajectory optimization code developed for the Global Trajectory Optimization Competition 13 (GTOC13) by the team **Yume Space**.  

This work does **not** reproduce the best-known solution achieved by the *School of Aerospace Engineering, Tsinghua University, Laboratory of Astrodynamics*, which reached a score of 345.216.  
However, this repository explores multiple optimization strategies for the GTOC13 problem and demonstrates how to reach a solution scoring around 50 points (not officially validated by the competition checker).


The repository compares several methods and is intended as an experimentation toolbox for the GTOC13 problem, and will later be complemented by a detailed guide explaining the implemented techniques and modeling choices.

The problem statement and associated data can be found on the following [website](https://gtoc.jpl.net/gtoc13/).

---

## Overview
This project covers several approaches, including:  

- Evolutionary algorithms  
- Reinforcement learning techniques  
- Direct single-shooting methods  
- Indirect optimal-control shooting

---

## Repository Structure
```
GTOC13-Trajectory-Optimization/   
│   
├── src/                 # Core source code (dynamics, optimization, utilities)   
├── missions/            # Problem setup files (GTOC13 configurations)   
├── notebooks/           # Jupyter notebooks for analysis and visualization   
├── results/             # Example outputs, plots, logs   
└── README.md            # Project documentation   
```

--- 

## Installation
Clone the repository:
```bash
git clone https://github.com/floriancal/GTOC13-Trajectory-Optimization.git'
python main.py
```
---

## Dependencies 
This project depends on the PyKEP library, released under the GPL license, which can be installed from the following [PyKep Documentation](https://esa.github.io/pykep/)


---


## 🤝 Contributions
Contributions, suggestions, and improvements are welcome.
Feel free to open an issue or submit a pull request.
