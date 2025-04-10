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