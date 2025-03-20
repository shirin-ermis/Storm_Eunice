# Storm Eunice attribution

This code supports the peer reviewed paper 

Ermis, Shirin, Nicholas J Leach, Fraser C Lott, Sarah N Sparrow, and Antje Weisheimer. ‘Event Attribution of a Midlatitude Windstorm Using Ensemble Weather Forecasts’. Environmental Research: Climate 3, no. 3 (1 September 2024): 035001. [https://doi.org/10.1088/2752-5295/ad4200](https://doi.org/10.1088/2752-5295/ad4200). 

## Research questions
- How was the extreme midlatitude windstorm Eunice (February 2022) impacted by climate change? How did the intensity of the gusts change with climate change?
- How dod the dynamics of the storm change with climate change? 

## Data availability
Post-processed data for the figures in the paper can be found on zenodo:
Ermis, S., & Leach, N. (2024). Storm Eunice (February 2022): Pre-industrial, current, and future climate scenarios using IFS EPS CY47R3 at 8, 4, and 2 days lead time [Data set]. In Environmental Research: Climate (1.0.0). Zenodo. [https://doi.org/10.5281/zenodo.10723245](https://doi.org/10.5281/zenodo.10723245).

For researchers in member states of the European Forecasting Centre ECMWF, the full data can be found on the MARS archhive under the following expeiment shorthands for the UK research
- b2nn, b2ns, b2nq for the preindustrial forecasts
- b2no, b2nr, b2nt for the future forecasts
- the current climate forecasts can be found in the operational archive.

Please see the folder ```mars_requests``` for examples on how to structure API requests to download this data.

## Code structure
Most of the analysis is in the folder ```notebooks``` along with the figures. Plot files are named after the number of the notebook that is used to create them. 
The ```stormeunice``` folder along with the ```setup.py``` create a mini-package that can be installed using pip and creates some data post-processing functions which are used in the data analysis.
An enviroment file can be found in the ```docs``` folder to create a virtual python environment.
The folder ```Nick``` contains early research contributions by Nicholas Leach (njleach) on this dataset.