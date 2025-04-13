# Project Implementation Documentation

## Environment Setup
To install the required libraries, I created a Python 3.10 virtual environment and executed:
```bash
make install
```

## Part I: Data Exploration and Model Development

After installing dependencies, I ran the [exploration notebook](../challenge/exploration.ipynb) and identified several issues:

1. The bar plots were not rendering correctly due to missing x and y specifications. I fixed this with:
   ```python
   sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values, alpha=0.9)
   ```

2. XGBoost was missing from the [requirements](../requirements.txt) file. I added it and ran:
   ```bash
   make install
   ```

### Model Comparison Analysis

| Metric              | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 | Model 6 | Model 7 |
|---------------------|---------|---------|---------|---------|---------|---------|---------|
| Precision (Class 0) | 0.81    | 0.82    | 0.88    | 0.81    | 0.88    | 0.81    | 0.81    |
| Recall (Class 0)    | 1       | 0.99    | 0.52    | 1.0     | 0.52    | 1.0     | 1.00    |
| F1 Score (Class 0)  | 0.9     | 0.90    | 0.66    | 0.9     | 0.65    | 0.9     | 0.90    |
| Precision (Class 1) | 0       | 0.56    | 0.25    | 0.76    | 0.25    | 0.53    | 0.53    |
| Recall (Class 1)    | 0       | 0.03    | 0.69    | 0.01    | 0.69    | 0.01    | 0.01    |
| F1 Score (Class 1)  | 0       | 0.06    | 0.37    | 0.01    | 0.36    | 0.03    | 0.03    |
| Accuracy            | 0.81    | 0.81    | 0.55    | 0.81    | 0.55    | 0.81    | 0.81    |

Since our primary objective is to detect delayed flights, recall and F1-score for Class 1 are critical metrics. Below is a ranking of models based on these criteria:

| Model   | F1 Score (Class 1) | Recall (Class 1) | Precision (Class 1) | Accuracy |
|---------|--------------------|-----------------:|--------------------:|----------|
| Model 3 | 0.37               | 0.69             | 0.25                | 0.55     |
| Model 5 | 0.36               | 0.69             | 0.25                | 0.55     |
| Model 2 | 0.06               | 0.03             | 0.56                | 0.81     |
| Model 6 | 0.03               | 0.01             | 0.53                | 0.81     |
| Model 7 | 0.03               | 0.01             | 0.53                | 0.81     |
| Model 4 | 0.01               | 0.01             | 0.76                | 0.81     |
| Model 1 | 0.00               | 0.00             | 0.00                | 0.81     |

Model 3 demonstrated the best performance for detecting delayed flights, which corresponds to:
```python
xgb_model_2 = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
```
I selected this model for deployment.

## Model Implementation

### Code Fixes
The IDE flagged an issue with type annotations:
```python
Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame)
```

I corrected it using square brackets as recommended:
```python
Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]
```

### Helper Functions
I created a [helper file](../challenge/utils.py) to organize methods and classes with proper documentation:
```python
get_period_day(date)
get_min_diff(data)
is_high_season(fecha)
```

I also implemented a scikit-learn pipeline to transform columns and created a [model directory](../model) to store the fitted pipeline and model.

When running the model tests for the first time:
```bash
make model-test
```

I encountered file path errors:
```
FAILED tests/model/test_model.py::TestModel::test_model_fit - FileNotFoundError: [Errno 2] No such file or directory: '../data/data.csv'
```

To resolve this, I added a DATA_PATH variable at the beginning of [test_model.py](../tests/model/test_model.py) and updated the data loading:
```python
self.data = pd.read_csv(filepath_or_buffer=f"{DATA_PATH}")
```

## Part II: API Development

For the model deployment with FastAPI, I created a [schemas file](../challenge/schemas.py) to define request and response models, and updated the predict endpoint in the [API file](../challenge/api.py).

Initial test execution with:
```bash
make api-test
```

Revealed compatibility issues:
```
FAILED tests/api/test_api.py::TestBatchPipeline::test_should_failed_unkown_column_1 - AttributeError: module 'anyio' has no attribute 'start_blocking_portal'
```

After research, I resolved this by updating the FastAPI dependencies:
```bash
pip install --upgrade fastapi starlette anyio
```

Subsequently, all four tests passed successfully. I then created the artifact in GCP's Artifact Registry and updated the URL in the Makefile to run tests from the GCP instance.

## Part III: Containerization and Deployment

I updated the [Dockerfile](../Dockerfile) and tested the API locally:
```bash
docker build -t challenge-api .
docker run -p 8080:8080 challenge-api
make stress-test
```

The initial stress test failed with:
```
ImportError: cannot import name 'escape' from 'jinja2'
```

I resolved this by upgrading Locust:
```bash
pip install --upgrade locust
```

For local API testing, I modified the `STRESS_URL = http://0.0.0.0:8080` in the [Makefile](../Makefile) and successfully passed all tests:

```
Response time percentiles (approximated)
Type     Name                                                            50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------|---------------------------------------------------------------|------|------|------|------|------|------|------|------|------|------|------|------
POST     /predict                                                        2300   3500   4200   4600   5100   5600   6900   8100  11000  11000  11000    604
--------|---------------------------------------------------------------|------|------|------|------|------|------|------|------|------|------|------|------
         Aggregated                                                      2300   3500   4200   4600   5100   5600   6900   8100  11000  11000  11000    604
```

## Part IV: CI/CD Implementation

I created the GitHub Actions workflow structure:
```bash
mkdir .github
cp -r workflows .github
```

For deployment, I set up a GCP project, created a new service account key, and added the JSON to GitHub Secrets as `GCP_CREDENTIALS`.

In the [workflows folder](../.github/workflows), I updated the CI/CD configuration files to leverage the commands provided in the [Makefile](../Makefile) for testing the deployment. The workflows are configured to trigger when changes are pushed to the main branch.

Since GitHub Actions defaults to Python 3.12, I explicitly specified Python 3.10 in the [CI workflow file](../.github/workflows/ci.yml):
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.10'
```
There was an issue when running the tests from the GCP instance, the error suggest
I should install httpx, I added it to the [test requirements](../requirements-test.txt).
```bash
  
  raise RuntimeError(
E   RuntimeError: The starlette.testclient module requires the httpx package to be installed.
E   You can install this with:
E       $ pip install httpx
```
### Dependency Resolution
During the CI workflow execution, I encountered a NumPy version incompatibility:
```
contourpy 1.3.1 requires numpy>=1.23, but you have numpy 1.22.4 which is incompatible.
```

I resolved this by updating the [requirements file](../requirements.txt):
```
numpy~=1.23
```

### Git Workflow Visualization
```
* 93a2e8e (HEAD -> dev) updating docs and api to solve cors issue
*   56cdebf (origin/dev) Merge tag '0.5.5' into dev
|\  
| *   08127dc (tag: 0.5.5, origin/main, main) Merge branch 'release/0.5.5'
| |\  
| |/  
|/|   
* | 8db5f56 updating url
* | 81c0685 Merge tag '0.5.4' into dev
|\| 
| *   efd56ee (tag: 0.5.4) Merge branch 'release/0.5.4'
| |\  
:...skipping...
* 93a2e8e (HEAD -> dev) updating docs and api to solve cors issue
*   56cdebf (origin/dev) Merge tag '0.5.5' into dev
|\  
| *   08127dc (tag: 0.5.5, origin/main, main) Merge branch 'release/0.5.5'
| |\  
| |/  
|/|   
* | 8db5f56 updating url
* | 81c0685 Merge tag '0.5.4' into dev
|\| 
| *   efd56ee (tag: 0.5.4) Merge branch 'release/0.5.4'
| |\  
| |/  
|/|   
* | 054ce55 updating url
* | 9863d1e Merge tag '0.5.3' into dev
|\| 
| *   0e1f423 (tag: 0.5.3) Merge branch 'release/0.5.3'
| |\  
| |/  
|/|   
* | 4028554 updating url
* | 2314ea2 Merge tag '0.5.2' into dev
|\| 
| *   ede489e (tag: 0.5.2) Merge branch 'release/0.5.2'
:...skipping...
* 93a2e8e (HEAD -> dev) updating docs and api to solve cors issue
*   56cdebf (origin/dev) Merge tag '0.5.5' into dev
|\  
| *   08127dc (tag: 0.5.5, origin/main, main) Merge branch 'release/0.5.5'
| |\  
| |/  
|/|   
* | 8db5f56 updating url
* | 81c0685 Merge tag '0.5.4' into dev
|\| 
| *   efd56ee (tag: 0.5.4) Merge branch 'release/0.5.4'
| |\  
| |/  
|/|   
* | 054ce55 updating url
* | 9863d1e Merge tag '0.5.3' into dev
|\| 
| *   0e1f423 (tag: 0.5.3) Merge branch 'release/0.5.3'
| |\  
| |/  
|/|   
* | 4028554 updating url
* | 2314ea2 Merge tag '0.5.2' into dev
|\| 
| *   ede489e (tag: 0.5.2) Merge branch 'release/0.5.2'
| |\  
| |/  
|/|   
| | * 8fee4bc (refs/stash) WIP on dev: 482e28a solving numpy dependency issue
| |/| 
|/| | 
| | * 3e3f91b index on dev: 482e28a solving numpy dependency issue
| |/  
|/|   
* | 482e28a solving numpy dependency issue
* | 070a5ff Merge tag '0.5.1' into dev

```
# Sending the Challenge
I created a [python script](../send.py) to send the challenge

```bash

python send.py
Status Code: 200
Response Text: {"status":"OK","detail":"your request was received"}
Success: The response matches the expected result.

```

# Conecting with front application

In order to connect the api with the front application that I made for the challenge I updated the
[api](../challenge/api.py). 