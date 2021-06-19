# urbana

Urban Data Analytics


## About the project

This project aims to perform a study of Barcelona using geographical data.

As an example, the targeted variable is the number of Airbnbs per census section. Other problems than can be tackled are the prices, the rating of the neighborhoods...

![Number of Airbnbs in Barcelona](https://github.com/egregorimar/urbana/blob/master/reports/figures/target.svg?raw=true)

For this task, the method used to exploit the spatial information is a **Geographically Weighted Regression (GWR)**.

This method allows to perform **local regressions** for each area taking in account its neighbours but not the whole dataset. This way, more subtle information can be captured.

![Error comparison](https://github.com/egregorimar/urbana/blob/master/reports/figures/error_comparison.svg?raw=true)

The results obtained by the GWR outperform the OLS. Both models used the features that were chosen in the feature selection stage (there were originally more than 80 features).

One of the advantages of GWR is that it allows to extract local information: the areas where a coefficient is significant and its value.

![Map of feature: POI_Souvenirs_Thrift_Store](https://github.com/egregorimar/urbana/blob/master/reports/figures/POI_Souvenirs_Thrift_Store.svg?raw=true)

![Map of feature: Percentage_Age_65_Plus](https://github.com/egregorimar/urbana/blob/master/reports/figures/Percentage_Age_65_Plus.svg?raw=true)

Furthermore, by applying a perturbative analysis to the linear model and taking the coefficients of the GWR for each area, both models can be compared:

![Model comparison: OLS vs GWR](https://github.com/egregorimar/urbana/blob/master/reports/figures/model_comparison.svg?raw=true)

## Development Setup

```sh
# Create conda environment, install dependencies on it and activate it
conda create --name urbana --file environment.yml
conda activate urbana

# Setup pre-commit and pre-push hooks
pre-commit install -t pre-commit
pre-commit install -t pre-push
```

## Update dependencies

To add new dependencies or to update existing ones:

1. Add the name (and version if needed) to the list of dependencies in `environment.yml`
2. run `conda env update --name urbana --file environment.yml  --prune`
3. Update the file `environment.lock.yml` by running `conda env export > environment.lock.yml`

## Credits

This package was created with the [BSCCNS/cookiecutter-data-science-conda](https://github.com/BSCCNS/cookiecutter-data-science-conda) project template.
