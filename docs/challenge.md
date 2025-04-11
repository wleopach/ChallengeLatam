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