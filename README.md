## Implementation Summary:
- network.py implements class for CPT's, DAG Validation, depth first search,
  topological sorting, pomegranate model export and scoring. 
- Learners.py implements the four learning algorithms
- generate_networks.py constructs all testing networks, it is called
  as needed by each analysis script

- Metrics.py implements methods to tabulate edge and variable predictions
  and to calculate summary statistics

## Network Analysis scripts
- Sampler_Accuracy.py reconstruct Icy Roads network from data and calculate
  KL divergence between observed and expected

- ds1_Analysis.py: run five trials for each learner against ds1

- ds2_Analysis.py: run five trials for each learner against ds2

- ds4_Analysis.py: run five trials for each learner against ds4

The number of iterations and the directory where results are saved can be
specified by changing the variables 'niter' and 'resdir' lines 11 - 13 of each file.
they are currently set to 10 for the sake of speed.  On biomix, 150 iterations over
ds4 takes ~ 4 hours, 300 iterations over ds4 takes ~ 10 hours.

## Candidate Selection Analysis scripts

- Time_CAS.py: time to construct candidate edge set (runs quickly)

- CAS_Edge_Overlap.py: print the edge intersection

 
## Key Parameters:
- For Laplace Approximation, alpha level was set to 0.00001
- Max 150 or 300 iterations, stop if 25 successive iterations with no imporvement.


## To run the simulation locally

- at the command prompt run: "python3 ds1_Analysis.py" (or which ever analysis is desired)

- To redirect standard output to a summary results file use: "python3  ds1_Analysis.py > result_summary.txt"




