# Environment preparation
 In order to install the libraries provided I created a virtual env with python 3.10 then I ran the 
```bash
    make install
```

# Part I

With the dependencies installed I proceeded to run the exploration [`notebook`](../challenge/exploration.ipynb). 
I found a problem with the bar plots, without x and y specified there was an issue.
```python
sns.barplot(x=flights_by_airline.index, y =flights_by_airline.values , alpha=0.9)

```
Since the xgboost was not added at the end of the [`requirements`](../requirements.txt), and ran
```bash
   make install 
```
again.

## 🔍 Model Comparison Summary

| Metric              | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 | Model 6 | Model 7 |
|---------------------|---------|---------|---------|---------|---------|---------|---------|
| Precision (Class 0) | 0.81    | 0.82    | 0.88    | 0.81    | 0.88    | 0.81    | 0.81    |
| Recall (Class 0)    | 1       | 0.99    | 0.52    | 1.0     | 0.52    | 1.0     | 1.00    |
| F1 Score (Class 0)  | 0.9     | 0.90    | 0.66    | 0.9     | 0.65    | 0.9     | 0.90    |
| Precision (Class 1) | 0       | 0.56    | 0.25    | 0.76    | 0.25    | 0.53    | 0.53    |
| Recall (Class 1)    | 0       | 0.03    | 0.69    | 0.01    | 0.69    | 0.01    | 0.01    |
| F1 Score (Class 1)  | 0       | 0.06    | 0.37    | 0.01    | 0.36    | 0.03    | 0.03    |
| accuracy            | 0.81    | 0.81    | 0.55    | 0.81    | 0.55    | 0.81    | 0.81    |

Since we want to detect delayed flights, recall and F1-score are critical, in the following table
the models are ordered according to this metrics: 

| Model | F1 Score (Class 1) | Recall (Class 1) | Precision (Class 1) | Accuracy |
|-------|--------------------|------------------|----------------------|----------|
| Model 3 | 0.37             | 0.69             | 0.25                 | 0.55     |
| Model 5 | 0.36             | 0.69             | 0.25                 | 0.55     |
| Model 2 | 0.06             | 0.03             | 0.56                 | 0.81     |
| Model 6 | 0.03             | 0.01             | 0.53                 | 0.81     |
| Model 7 | 0.03             | 0.01             | 0.53                 | 0.81     |
| Model 4 | 0.01             | 0.01             | 0.76                 | 0.81     |
| Model 1 | 0.00             | 0.00             | 0.00                 | 0.81     |

The model that better detects delayed flights is Model 3, this corresponds to:
```python
xgb_model_2 = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
```
I will use this model for the deployment.

## Model transcription 📝

The IDE highlighted this in red,
```python
Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame)
```
I changed to square brackets as suggested
```python
Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]
```
### Helper functions
I created a [`helper`](../challenge/utils.py) file with functions to store some methods and classes,
I used the functions from the [`notebook`](../challenge/exploration.ipynb) with documentation:
```python
get_period_day(date)
get_min_diff(data)
is_high_season(fecha)
``` 
I added a sklearn pipeline to transform the columns, and a [`model`](../model) folder where the fitted pipeline is going to be stored,
and the fitted model as well.

When executing 
```bash
 make model-test
```
for the first time I got this error,
```bash
FAILED tests/model/test_model.py::TestModel::test_model_fit - FileNotFoundError: [Errno 2] No such file or directory: '../data/data.csv'
FAILED tests/model/test_model.py::TestModel::test_model_predict - FileNotFoundError: [Errno 2] No such file or directory: '../data/data.csv'
FAILED tests/model/test_model.py::TestModel::test_model_preprocess_for_serving - FileNotFoundError: [Errno 2] No such file or directory: '../data/data.csv'
FAILED tests/model/test_model.py::TestModel::test_model_preprocess_for_training - FileNotFoundError: [Errno 2] No such file or directory: '../data/data.csv'

```
In order to solve this issue I added the DATA_PATH variable at the beginning of the  [`test_model`](../tests/model/test_model.py)
and then read the data using:
```python
self.data = pd.read_csv(filepath_or_buffer=f"{DATA_PATH}")
```

# Part II 🧨

In order to deploy the model using fastapi I created a [`schemas`](../challenge/schemas.py) file to define the 
response and requests models. I updated the predict endpoint in [`api`](../challenge/api.py). 
The first time that I ran
```bash
 make api-test
```
I got the following errors:
```bash
=================================================================================================== short test summary info ===================================================================================================
FAILED tests/api/test_api.py::TestBatchPipeline::test_should_failed_unkown_column_1 - AttributeError: module 'anyio' has no attribute 'start_blocking_portal'
FAILED tests/api/test_api.py::TestBatchPipeline::test_should_failed_unkown_column_2 - AttributeError: module 'anyio' has no attribute 'start_blocking_portal'
FAILED tests/api/test_api.py::TestBatchPipeline::test_should_failed_unkown_column_3 - AttributeError: module 'anyio' has no attribute 'start_blocking_portal'
FAILED tests/api/test_api.py::TestBatchPipeline::test_should_get_predict - AttributeError: module 'anyio' has no attribute 'start_blocking_portal'

```
After doing some research I found that the issue could be solved by doing an update of the fastapi, startlette and
anyo versions. I did

```bash
pip install --upgrade fastapi starlette anyio
```
After this all 4 tests passed.

# Part III 🚀

For the deployment I updated the [`Dockerfile`](../Dockerfile), and tested the API locally.

```bash
  docker build -t challenge-api .
  docker run -p 8080:8080 challenge-api
  make stress-test
```
The first time I did the stress-test I got this error:
```bash
# change stress url to your deployed app 
mkdir reports || true
locust -f tests/stress/api_stress.py --print-stats --html reports/stress-test.html --run-time 60s --headless --users 100 --spawn-rate 1 -H http://127.0.0.1:8000 
mkdir: cannot create directory ‘reports’: File exists
Traceback (most recent call last):
  File "/home/leonardo/PersonalProjects/ChallengeLatam/.venv/bin/locust", line 5, in <module>
    from locust.main import main
  File "/home/leonardo/PersonalProjects/ChallengeLatam/.venv/lib/python3.10/site-packages/locust/main.py", line 16, in <module>
    from .env import Environment
  File "/home/leonardo/PersonalProjects/ChallengeLatam/.venv/lib/python3.10/site-packages/locust/env.py", line 5, in <module>
    from .web import WebUI
  File "/home/leonardo/PersonalProjects/ChallengeLatam/.venv/lib/python3.10/site-packages/locust/web.py", line 14, in <module>
    from flask import Flask, make_response, jsonify, render_template, request, send_file
  File "/home/leonardo/PersonalProjects/ChallengeLatam/.venv/lib/python3.10/site-packages/flask/__init__.py", line 14, in <module>
    from jinja2 import escape
ImportError: cannot import name 'escape' from 'jinja2' (/home/leonardo/PersonalProjects/ChallengeLatam/.venv/lib/python3.10/site-packages/jinja2/__init__.py)
make: *** [Makefile:29: stress-test] Error 1

```
After doing some research I found that updating locust was enough to resolve the issue:
```bash
  pip install --upgrade locust
```

In order to test the api locally I changed the STRESS_URL =  http://0.0.0.0:8080 [`Makefile`](../Makefile)  and passed
all the tests:

```bash


Response time percentiles (approximated)
Type     Name                                                                                  50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------|--------------------------------------------------------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
POST     /predict                                                                             2300   3500   4200   4600   5100   5600   6900   8100  11000  11000  11000    604
--------|--------------------------------------------------------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
         Aggregated                                                                           2300   3500   4200   4600   5100   5600   6900   8100  11000  11000  11000    604

```

# Part IV ❄

I created the folder .github and copied the workflows folder inside it.
```bash
    mkdir .github
    cp -r workflows .github
```

I used  GCP for the project deployment, I created a new key to a project on GCP 
and added the JSON to github secrets as GCP_CREDENTIALS.

In the  [`workflows`](../.github/workflows) folder I updated the cd.yml and ci.yml, and added the
files to the project. This workflows will activate when a push to main branch is executed.

I configured the workflows so that the  provided commands in the [`Makefile`](../Makefile) are used to test the 
deployment.

By default the github actions runs with python3.12, so I had to explicitly use python 10 in the 
[`ci-workflow`](../.github/workflows/ci.yml).
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.10'
```
