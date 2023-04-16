<h1 align="center" style="color:MediumSeaGreen;"> <b>  Prediction-with-skLearn</b>

# Objective
Using pycaret model outcomes to aid in building an skLearn model for analyzing Airbnb data in European cities.

# Technologies Used

* [skLearn](https://scikit-learn.org/stable/index.html)
* [pandas](https://pandas.pydata.org/)
* [seaborn](https://scikit-learn.org/stable/index.html)
* [Yellowbrick](https://www.scikit-yb.org/en/latest/)
# Data
The information utilized in the project was the data preprocessed by pycaret.
```
data = pd.read_csv('C:/Users/piett/Intro Pycaret/Pycaret3/data/euBnB_transformed.csv')
print(data.shape)
data
```
# Data setup
Excluding data to be utilized as unseen data for making predictions. 
```
data_unseen = data.sample(frac=0.1)
data = data.drop(data_unseen.index)
print(f'Data for model: {data.shape}, \nData for unseen predicitions: {data_unseen.shape}')
data_unseen.to_csv('C:/Users/piett/Intro Pycaret/Pycaret3/data/euBnB_unseenX.csv', index=False)
```

The main setup required was to set X and y values.
```
X = data.drop(['cost per night cad'], axis= 1)
y = data['cost per night cad']
```
# Model testing
The subsequent model employed was derived from pycaret.

<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/HyperParameters.png" align="center">

----

Take note that an error may arise when copying any model hyperparameters from pycaret, but this can be resolved by merely eliminating the 'nan' value.

<img src="https://github.com/Piettro314/Prediction-with-skLearn/blob/main/media%20content/nan%20error.png" align="center">


The subsequent code was executed to assess the models:

```
%%time

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=42)


models = [
XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=12, max_leaves=None,
             min_child_weight=3, monotone_constraints=None,
             n_estimators=600, n_jobs=-1, num_parallel_tree=None,
             predictor=None, random_state=None)
]


def score_model(X, y, estimator, **kwargs):
    """
    Test various estimators.
    """
    model = Pipeline([
        ('one_hot_encoder', OneHotEncoder()),
        ('estimator', estimator)
    ])

    # Instantiate the classification model and visualizer
    model.fit(X, y, **kwargs)

    expected  = y
    predicted = model.predict(X)

    # Compute and return F1 (harmonic mean of precision and recall)
    print(f"{estimator.__class__.__name__}:\n R_sq: {r2_score(expected, predicted):.4}\n"
          f" MAE: {mean_absolute_error(expected, predicted):,.2f}\n"
         )

for model in models:
    score_model(X, y, model)
```
The purpose of the aforementioned code is to perform a train-test split on the chosen model and to obtain its R-squared and mean absolute error (MAE) scores. Hyperparameters were tuned until a satisfactory R-squared and MAE were attained.

<img src="https://github.com/Piettro314/Prediction-with-skLearn/blob/main/media%20content/resultsAfterIterations.png" align="center">

# Visualizations
Results from model then observed using the following charts.

<img src="https://github.com/Piettro314/Prediction-with-skLearn/blob/main/media%20content/feature%20importance%20sk.png" align="center">

<img src="https://github.com/Piettro314/Prediction-with-skLearn/blob/main/media%20content/Residual%20sk.png" align="center">

<img src="https://github.com/Piettro314/Prediction-with-skLearn/blob/main/media%20content/Prediction%20Error.png" align="center">

# Predict the model
Evaluating the model's performance by testing it against unseen data.

```
pred = model.predict(X=data_to_pred)
pred

pred_results = data_unseen.loc[:,['cost per night cad']]
pred_results['y_pred'] = pred
pred_results['Percent Diff'] = (pred_results['cost per night cad']-pred_results['y_pred'])/pred_results['cost per night cad']
pred_results
```

<a href="https://github.com/Piettro314/Prediction-with-skLearn"><img src="https://github.com/Piettro314/Prediction-with-skLearn/blob/main/media%20content/results%20against%20Unseen.png" align="center" /></a>

Upon analyzing the outcome, it was noted that the mean discrepancy between the projected and actual cost, when compared to unseen data, was -1.05%.