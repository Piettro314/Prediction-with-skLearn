<h1 align="center" style="color:MediumSeaGreen;"> <b>  Prediction-with-skLearn</b>

# Objective
Using pycaret model outcomes to aid in building an skLearn model for analyzing Airbnb data in European cities.

# Technologies Used

* [skLearn](https://scikit-learn.org/stable/index.html)
* [pandas](https://pandas.pydata.org/)
* [seaborn](https://scikit-learn.org/stable/index.html)
* [Yellowbrick](https://www.scikit-yb.org/en/latest/)

# Data setup

The main setup required was to set X and y values.
```
X = data.drop(['cost per night cad'], axis= 1)
y = data['cost per night cad']
```
# Model testing
Coming from pycaret the following model was used

<img src="https://github.com/Piettro314/Prediction-with-Pycaret/blob/main/media%20content/HyperParameters.png" align="center">

The following code was ran:

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
What the above code accomplishes is to do a train test split against the selected model giving R_sq and MAE score. Hyper parameters were adjusted until a desirable R_sq and MAE was achieved

