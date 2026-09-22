# PWFluctuations

Code and numerical data accompanying the manuscript

**"Physical-Work Fluctuation Relations from Accessible Quantum Macrostates"**

by Borhan Ahmadi.

## Overview

This repository contains the standalone numerical code and validation data used to generate the figures, tables, and numerical results reported in the manuscript and Supplemental Material.

The main calculation considers a finite Bose--Hubbard system and evaluates the physical-work-and-record fluctuation relation, the common-target free-energy control-variate construction, and the associated information--sampling frontier.

## Main files

- `FIGURES_TABLES_FREE_ENERGY_STANDALONE.py`  
  Main standalone script. It computes the nominal Bose--Hubbard dynamics, endpoint maximum-entropy state, fluctuation-relation quantities, free-energy sampling analysis, and the figures and tables used in the manuscript.

- `arbitrary_initial_state_validation_STANDALONE.py`  
  Independent validation script for the arbitrary-initial-state extension and related numerical checks.

- `input_data/`  
  Contains the validated numerical data used for the Supplemental-Material checks.

- `requirements.txt`  
  Lists the required Python packages.

## Requirements

Python 3 with the following packages:

```text
numpy
scipy
pandas
matplotlib
