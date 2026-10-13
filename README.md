# Fluctuation of Urban Water Demand Follows Time-Variant Power Law Scaling

<p align="center">
  <image src = "README_resources\GA_Fluctuation.png" width="800"/>
</p>

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Feedback](#feedback)
- [Contributors](#contributors)
- [Installation](#installation)
- [License](#license)
- [Notes](#notes)

## Introduction
[(Back to top)](#table-of-contents)

This repository contains all code for the paper **Fluctuation of Urban Water Demand Follows Time-Variant Power Law Scaling** by Sattler, B.J.; Cheong, S.A.; Joerin, J. and Pelz, P.F..

The paper shows that the daily sample variance of district metered area (DMA)-level urban water demand scales according to a power law of the daily sample mean.
Power law scaling of the variance to the mean is called Taylor's Law (TL).
The paper shows that the parameters of this scaling are temporally stable but vary over longer time horizons.
Stable regimes persist for multiple weeks and could therefore be leveraged for a better understanding and prediction of water demand fluctuation.

To detect the regimes, the code first calculates the mean and variance of the daily samples, i.e., groups of 24 hourly water meter readings.
Next, a particle filter is used to estimate the parameters of the power law scaling given by $s^2=\alpha m^\beta$ and the observational uncertainty.
Finally, the results are plotted and summarized by the scripts.

## Dependencies

Detailed information on dependencies and the versions used for the generation of the results in the paper is given in the `environment.yml` file.

- Python 3.14

1. pandas:              data import, data handling
2. numpy:               data handling and vectorization of operations
3. h5py:                data import, data export
4. scipy:               methods used for sampling and resampling
5. matplotlib:          plotting

## Contained files

### Scripts:
All following scripts are located in the `\src\time_variant_power_law_fluctuation_scaling` directory.

1. `__main__.py`:         Executable main script.
Will run the scripts 2 through 6 and produce the plots and Table contents for the paper.
2. `data_cleaning.py`:    Methods for preprocessing the input data.
Cleans empty, incomplete or duplicate data, averaging to hourly data for the WSND data (inflow.csv).
3. `PF_type_estimate.py`: Methods for estimating the TL parameters for the time series as explained in the paper.
4. `results_analysis.py`: Methods to aggregate the Monte Carlo runs and reduce the results to the point estimates and credible intervals.
5. `plotters.py`:         Methods to produce the plots displayed in the paper.
6. `generate_table_2.py`: Methods to generate the $\LaTeX$ code for Table 2 of the paper.
7. `dlr_style.mplstyle`:  Matplotlib configuration file for correct representation of plots.

### Input data:
The input data is available under:
https://datastore.dlr-pi.de/s/pFnNTzCBgrFYCW7

Please download the two input files and place them in the `\src\time_variant_power_law_fluctuation_scaling`  directory to enable the data import.
Both files are NOT part of the work in this paper but are provided in the following publications:


1. `Inflow_Data_4.xlsx`:  Data from the Battle of Water Demand Forecasting (BWDF).
Alvisi, Stefano, et al. "Battle of water demand forecasting." Journal of Water Resources Planning and Management 151.10 (2025): 04025049, https://doi.org/10.1061/JWRMD5.WRENG-6887.
2. `inflow.csv`:          Water and Sewer Network Data (WSND).
Mohan Doss, P., et al. “Sewer Network and Smart Water Meter Data for Modelling and Analysis of Water Distribution and Sewer Networks”. Version 1.0.0, Zenodo, 29 Oct. 2024, https://doi.org/10.5281/zenodo.14001027.

### Dependency file:
1. `environment.yml`:     Packages and versions necessary to run the code.

## Features
[(Back to top)](#table-of-contents)

The code can be run by executing the main script via the terminal input `python __main__.py`.
This generates the results presented in the paper.
The code performs the following steps of analysis:

### Data import, cleaning, preprocessing

1. Data import: Imports BWDF data from the `Inflow_Data_4.xlsx` file and WSND data from the `inflow.csv` file.

2. Data cleaning and preprocessing:
Removes all duplicate or incomplete data, aggregates WSND data to hourly time stamps, and generates the daily sample mean and variance.
Transforms all data into logarithmic space to aid subsequent estimation tasks.

### Particle Filter
The steps 3. to 5. are performed multiple times to obtain a Monte Carlo set of estimates.
Furthermore, they are performed both for a free estimation of $\beta$ and for a alternative constrained model of $\beta =2$.

3. Prior definition & initialization:
Defines prior ranges for the scaling exponent $\beta \in [0,50]$, observation noise scale $\log(\tau) \in [-6, -1]$ and calculates matching data-driven priors for the TL intercept $\log(\alpha)$ based on the first observation.
Generates $N$ particles in this prior range using Latin hypercube sampling and initializes all weights to $1/N$.

4. Updating:
Calculates the orthogonal distance of the current observation from the TL line represented by each particle and evaluates the particles' likelihood under a Gaussian $\mathcal{N}(0,\tau^2)$.
Updates each particle's weight accordingly.
This step is repeated for each subsequent observation of a time series.

5. Resampling:
Is performed if the effective sample size falls below a threshold: $\sum\frac{1}{w^2} \leq 0.2*N$. Approximates the posterior through a weighted Gaussian kernel density estimate of the particles and weights, and draws $N$ new samples.
Replaces $0.5\%$ of the resampled particles by the same number of particles generated by the same process as the initialization described in step 3 with the current observation.

### Results analysis

6. Result aggregation:
Step 6 combines the posterior weights and particles for all Monte Carlo runs to obtain a joint estimate.

7. Result analysis:
Reduces the set of Monte Carlo estimates to the marginal medians and marginal credible intervals for each parameter.

8. Exports of plots and result table:
The obtained results are plotted to generate the figures displayed in the paper and are exported to generate Table 2 of the paper.

## Feedback
[(Back to top)](#table-of-contents)

Bernhard Jonathan Sattler, Institute for the Protection of Terrestrial Infrastructures, German Aerospace Center (DLR), DLR-PI, bernhard.sattler@dlr.de, https://www.dlr.de/en/pi

## Contributors
[(Back to top)](#table-of-contents)

- Bernhard Jonathan Sattler, Institute for the Protection of Terrestrial Infrastructures, German Aerospace Center (DLR)

## License
[(Back to top)](#table-of-contents)

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/). Please have a look at the [LICENSE.txt](LICENSE.txt) for more details.

## Notes
[(Back to top)](#table-of-contents)

The analysis was performed using an Intel i9-10980XE processor with 128GB RAM.
Including intermediate results generated during the analysis, around 60GB of permanent storage are needed.
Machines with significantly lower computational power might run into issues executing the code.
To reduce RAM requirements, the variables `common_particle_count`, `common_mc_runs`, and `common_batches`  in the `__main__.py` can be reduced.
This might impact the accuracy of the obtained results.


<p align="center">
  <image src="https://www.dlr.de/apple-touch-icon.png" width="180"/>
</p>
