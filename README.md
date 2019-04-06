## Adaptive-Soft-Sensor-Design
Github repo for the research paper titled "Integrating Adaptive Moving Window and Just-in-Time Learning Paradigms for Soft-Sensor Design"

# 1. Matlab Code 
- Please cite the corresponding paper if you use the code, and do let me know if you encounter any errors/bugs!! I'd be more than glad to help debug. (note to self: add .bib entry when paper gets published)

- I tried my best to clean up the code. But keep in mind that the code has gone through countless changes during my MS thesis, so you might see a few unnecessary variables defined here and there. Any feedback on how to improve the program and make it run faster is appreciated :)
- Our method is based on relevance vector machine (RVM), I used the [SparseBayes software version 2.0] (http://www.miketipping.com/downloads.htm), provided by Tipping himself. I had to modify the code in *SB2_FullStatistics.m* at line 105 to make sure that matrix A is positive definite. I used [nearestSPD.m](https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd) to get the nearest PD matrix.
- You can find our implementation of MWAdp and MWAdpJITL methods on [debutanizer column data](https://www.springer.com/gp/book/9781846284793) in *Demo.m* file.

# 2. Simulation Data
20 simulations runs were conducted for 8 different concept drift models (CDM). Details of the simulation models can be found in the article. Each CDM is stored as a struct with "X" and "Y" arrays. For a total of 700 observations, 19 predictors and their 2 lagged measurements are included in X (there are 19x3 = 57 predictors/columns in total in X) and the response variables (concentration of product B) is in Y.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2619947.svg)](https://doi.org/10.5281/zenodo.2619947)

