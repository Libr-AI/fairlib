<!-- Model training -->
- Implementations of a list of debiasing approaches
- Easy to use API: can be imported as a package in other projects or run form the command line.
- Ability to train new models on new data with existing debiasing approaches, including CV and ML classification tasks.
- Easy to track and reproduce experiments via YAML configuration files.
<!-- Evaluation -->
- Implementations of confusion-matrix based fairness metrics, such as equal opportunity, equalized odds, and demographic parity.
- Implementations of DTO, constrained selection, and Pareto frontier for model selection.
- Evaluation and model selection API: automatically process in parallel.
<!-- Analysis -->
- Analysis API: merge results for different methods as a single object, which can be used for creating latex tables and trade-off plots.
- Ability to save configurations of reported models.
- Experimental results outperform original scores.