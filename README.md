# urbana

Urban Data Analytics

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
