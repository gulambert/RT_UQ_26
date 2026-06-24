In this repo, small pieces of code to introduce our GP class and acquisition function Jacques applied to classical test cases.

Here are some requirements :

- python >= 3.11
- botorch >= 0.17.2
- gpytorch == 1.15.2
- scipy==1.17.1
- torch==2.11.0

You then have :

- GP class that is build upon GPyTorch and Botorch librairies and naturally benefits tensorization / GPU use. 

- Acquisition function : Jacques for Jacobian-based Active Learning presented during the 2026 session of MASCOT NUM days.

- A notebook to get through the methodology (first on Branin, and on the multi-output case (Ishigami, Branin).

This repo is a subset of the future ActiveDGSM repo linked to the paper : 

"Gradient-based Active Learning with Gaussian Processes for Global Sensitivity Analysis"

Have a good day !
