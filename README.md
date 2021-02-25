# NHS-PROMs
A data science case study of the NHS Digital PROMs data of hip and knee replacements

## Conda environment 
### Create
- Clone this git repository: `git clone <git url>`
- Start your conda prompt and go to your local repository folder: `cd <path to local repository>`
- Create a new environment named "jads" from the YML-file: `conda env create --file environment.yml`
- Activate the new environment: `conda activate jads`
- Start jupyter: `jupter notebook`
    
### Update (e.g. after new requirements)
There are two ways possible:
- From your active environment: `conda env update --file conda_env.yml`
- If your environment is not activated: `conda env update --name jads --file environment.yml`

### Remove
- Make sure that environment is not active. Otherwise, deactivate it: `conda deactivate`
- Remove environment: `conda env remove --name jads`