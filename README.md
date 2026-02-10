![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
# GTOC13-Trajectory-Optimization

This repository contains trajectory optimization code developed for the Global Trajectory Optimization Competition 13 (GTOC13) by the team **Yume Space**.  

This work does **not** reproduce the best-known solution achieved by the *School of Aerospace Engineering, Tsinghua University, Laboratory of Astrodynamics*, which reached a score of 345.216.  

However, this repository explores 3 optimization strategies (for the moment only Method 1 is available) for the GTOC13 problem and demonstrates how to reach a solution scoring around 75 points with the so called 'Method 1' shown hereunder.


The repository compares several methods and is intended as an experimentation toolbox for the GTOC13 problem, it will hopefully be complemented by a detailed guide explaining the implemented techniques and modeling choices.

The problem statement and associated data can be found on the following [website](https://gtoc.jpl.net/gtoc13/).

The solution outputed from this script can be validated against the checker available on the website (an account is needed).


---

## Overview

This project covers several approaches divided in 'methods', including:  

- Method 1 : Classical Minimization techniques --> Nelder-Mead / L-BFGS-B (heavily relies on scipy.minimize lib)
- Method 2 :Evolutionary algorithms  
- Method 3 :Reinforcement learning techniques 

As of today only Method 1 is available on this repo  ! Work in progress :) ! 

For the solar sail arcs solving :
- Direct single-shooting methods  
- Indirect optimal-control shooting
! Work in progress :) ! 


---

## Repository Structure
```
GTOC13-Trajectory-Optimization/   
│   
├── data/                # Problem setup files (GTOC13 configurations)   
├── doc/		 # Methods description (Markdown format)  
├── results/             # Example outputs, plots, logs   
├── src/                 # Core source code 
└── README.md            

```

--- 

## Results 
Under results/ folder are listed some obtained solution. The submission.txt file is the file format requested by the GTOC13 checker and the file has been validated against it. 
The html file contains a 3D animation of the solutions (do not hesitate to zoom and pan while running the sim but avoid rotational button as they usually makes things messy to watch). 

The 75 points solution html file is [here](https://htmlpreview.github.io/?https://github.com/floriancal/GTOC13-Trajectory-Optimization/tree/main/results/Method_1_J75/Simulation.html) 

## Installation and use

Clone the repository:
```bash
git clone https://github.com/floriancal/GTOC13-Trajectory-Optimization.git'
python main.py
```

Do not hesitate to open main.py to play with the available options (method selection will be placed here in the future).

---

## Dependencies 
This project depends on the PyKEP library, released under the GPL license, which can be installed from the following [PyKep Documentation](https://esa.github.io/pykep/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2575462.svg)](https://doi.org/10.5281/zenodo.2575462)
Dario Izzo. (2019). esa/pykep: Bug fixes and more support on Equinoctial Elements (v2.3). Zenodo. https://doi.org/10.5281/zenodo.2575462

Distlink available [here](https://github.com/maxiimilian/distlink)

---


## 🤝 Contributions
Contributions, suggestions, and improvements are welcome.
Feel free to open an issue or submit a pull request.
