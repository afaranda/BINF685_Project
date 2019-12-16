## Code Summary:
- main.py runs the simulation and tabulates data
- network.py implements class for CPT's, DAG Validation and model training

## Key Parameters:
- For Laplace Approximation, alpha levels of 1 and 0.001 were tested
- Max 300 iterations, or 50 iterations with no imporvement.


## The following files should be placed in the same directory.
- AF_HW2_Runner.sh
- hw2_train.data
- hw2_test.data
- main.py
- network.py
- Plot_Train.R  (Optional to plot learning rate)

## To run the simulation on biomix:

In the working directory, run: "sbatch AF_HW2_Runner.sh"

## To run the simulation locally

- at the command prompt run: "python3 main.py"

- To redirect standard output to a summary results file use: "python3  main.py > result_summary.txt"

- To generate plot: "R CMD BATCH Plot_Training.R"

