# McMLP (Metabolic response predictor using coupled Multilayer Perceptrons)
This repository contains scripts needed to run `McMLP` that is designed to predict metabolomic profiles after the dietary intervention (i.e., endpoint metabolomic profiles) based on the baseline data (such as the microbial compositions and metabolomic profiles before the dietary intervention) and the dietary intervention strategy. A preprint that describes the method in detail can be found [here](). 
![schematic](schematic.png)

## Versions
The version of Python we used is 3.7.3.

## Dependencies
Necessary Python packages can be found in `requirements.txt`. Installing those packages can be achieved by pip:
```
pip install -r requirements.txt
```

The entire installation process takes less than half an hour for the setting we used (Macbook Air M1 2020, 8GB RAM, 512GB SSD).

## Workflow
1. **Data processing**: we apply the CLR (Centred Log-Ratio) transformation to both tbe microbiota and metabolomic profiles. The dietary intervention strategy is encoded as a binary vector. An example of the processed data of avocado intervention is saved in the folder `./data/avocado_SCFAs`.
2. **McMLP**: the Python script `McMLP.py` loads the processed data in `./data/avocado_SCFAs/processed_data`. It starts the training of McMLP on the training data and then makes the final predictions for the endpoint metabolomic profiles. The final predicted CLR-transformed metabolomic profiles are saved as `./results/predicted_metabolomic_profiles.csv`, the true values of CLR-transformed metabolomic profiles are saved as `./results/true_metabolomic_profiles.csv`, and the Spearman Correlation Coefficients for all metabolites are saved as `./results/metabolites_corr.csv`.
3. **Inferring microbe-metabolite interactions**: the Python script `McMLP_inferring_interactions.py` loads the processed data in `./data/avocado_SCFAs/processed_data` and applied the sensitivity analysis to the trained McMLP to capture how microbial relative abundances respond to the perturbation in the intervened food and how the metabolite concentrations respond to the perturbation in the microbial relative abundances. All values of sensitivities of microbes towards foods are saved as `./results/sensitivity_diet_and_microbes.csv`. All values of sensitivities of metabolites towards microbes are saved as `./results/sensitivity_diet_and_microbes.csv`.

## Example
We showed an example of training McMLP on the avocado dataset. The command for training and the following prediction is:
```
<PATH_TO_PYTHON> ./McMLP.py NUM_OF_SPLITS PATH_TO_PROCESSED_DATA IF_BASELINE_METABOLOME_INCLUDED
```
<PATH_TO_PYTHON> is the path to the executable Python file located under the installed folder, NUM_OF_SPLITS is the number of train-test splits, PATH_TO_PROCESSED_DATA is the path to the processed data that we want to load to McMLP, and IF_BASELINE_METABOLOME_INCLUDED is a boolean variable to denote whether we would like to include the baseline metabolomic profiles in the input. For the avocado dataset that we demonstrate here, the command for running five train-test splits with the baseline metabolomic profiles is
```
python ./McMLP.py 5 "./data/avocado_SCFAs/processed_data/" True
```
Similarly, to infer interactions via the sensitivity method, we can run `McMLP_inferring_interactions.py`:
 ```
<PATH_TO_PYTHON> ./McMLP_inferring_interactions.py PATH_TO_PROCESSED_DATA IF_BASELINE_METABOLOME_INCLUDED
```
The command on the inference with baseline metabolomic profiles on the real dataset is 
```
python ./McMLP_inferring_interactions.py "./data/avocado_SCFAs/processed_data/" True
```
It is also possible to train McMLP in Jupyter notebooks. One example is provided as `McMLP_runner.ipynb`. The entire running process takes less than 20 minutes for the setting we used (Macbook Air M1 2020, 8GB RAM, 512GB SSD).

## License

This project is covered under the **MIT License**.