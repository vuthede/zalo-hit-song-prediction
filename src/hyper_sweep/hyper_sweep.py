%load_ext autoreload
%autoreload 2
# https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a
# https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Bayesian%20Hyperparameter%20Optimization%20of%20Gradient%20Boosting%20Machine.ipynb
from hyperopt import hp
import numpy as np
import lightgbm as lgb
from hyperopt import STATUS_OK
import hyperopt 
from hyperopt import tpe
from hyperopt import Trials
import csv
import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
import ray
from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.logger import DEFAULT_LOGGERS

from baseline5_for_sweep import perform_cv_lightgbm, get_training_set


# Define the search space

space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type',
                               [{'boosting_type': 'gbdt',
                                 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                {'boosting_type': 'dart',
                                 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                {'boosting_type': 'goss'}]),
    'num_leaves': hp.choice('num_leaves', np.arange(30, 151, dtype=int)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.1)),
    'subsample_for_bin': hp.choice('subsample_for_bin', np.arange(1, 16, dtype=int)),
    'min_child_samples': hp.choice('min_child_examples', np.arange(4, 101, dtype=int)),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'seed': 99999,
    'bagging_freq': hp.choice('bagging_freq',
                              [
                                  {'gbdt': hp.choice('gbdt_bagging_freq', np.arange(1, 25, dtype=int))},
                              ]),
    'bagging_fraction': hp.choice('bagging_fraction',
                                  [
                                      {'gbdt': hp.uniform('gbdt_bagging_fraction', 0.95, 1)},
                                  ]),
    'boost_from_average': 'false',
    'feature_fraction': hp.uniform('feature_fraction', 0.05, 1),
    'max_depth': -1,
    'metric': 'root_mean_squared_error',
    'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(5, 20, dtype=int)),
    'num_threads': 1,
    'tree_learner': 'serial',
    'objective': 'regression',
    'use_missing': True,
    'verbosity': 1,

}

# Sample from the full space
example = hyperopt.pyll.stochastic.sample(space)

# Dictionary get method with default
subsample = example['boosting_type'].get('subsample', 1.0)

# Assign top-level keys
example['boosting_type'] = example['boosting_type']['boosting_type']
example['bagging_fraction'] = example['bagging_fraction'].get(example['boosting_type'], 1)  # default to 1 unless lgtb
example['bagging_freq'] = example['bagging_freq'].get(example['boosting_type'], 0)  # default to 0 unless lgtb
example['subsample'] = subsample


def objective(params, reporter):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['bagging_fraction'] = params['bagging_fraction'].get(params['boosting_type'], 1)  # default to 1 unless lgtb
    params['bagging_freq'] = params['bagging_freq'].get(params['boosting_type'], 0)  # default to 0 unless lgtb
    params['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])

    start = timer()

    # Perform n_folds cross validation
    cv_results = perform_cv_lightgbm(df_train,
                                     chosen_features,
                                     params=params,
                                     early_stopping_rounds=100,
                                     n_folds=5)
    # lgb.cv(params, train_set, num_boost_round = 10000, nfold = n_folds,
    #                    early_stopping_rounds = 100, metrics = 'auc', seed = 50)

    run_time = timer() - start

    # Loss must be minimized
    loss = cv_results['rmse_mean']

    # Write to the csv file ('a' means append)
    # of_connection = open(out_file, 'a')
    # writer = csv.writer(of_connection)
    # writer.writerow([loss, params, ITERATION,
    #                 cv_results['rmse_std'],
    #                 cv_results['cv_stopping_iters'],
    #                 cv_results['best_stopping_iter_std'],
    #                 cv_results['best_stopping_iter_std'],
    #                 run_time])

    # Ray will negate this by itself to feed into HyperOpt
    reporter(timesteps_total=1, mean_loss=cv_results['rmse_std'])

    # Dictionary with information for evaluation
    return {'loss': loss,
            'params': params,
            'iteration': ITERATION,
            'rmse_std': cv_results['rmse_std'],
            'best_stopping_iter': cv_results['cv_stopping_iters'],
            'best_stopping_iter_mean': cv_results['best_stopping_iter_std'],
            'best_stopping_iter_std': cv_results['best_stopping_iter_std'],
            'train_time': run_time,
            'status': STATUS_OK}
df_train, chosen_features = get_training_set()


# Global variable
global ITERATION

ITERATION = 0

# https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/hyperopt_example.py
config = {
    "num_samples": 100,
    "config": {
        "iterations": 100,  # passed to our objective --> no effect I think
    },
    "stop": {
        "timesteps_total": 999 # Passed to our objective --> no effect I think but can be used to stop a job early I think.
    },
}

algo = HyperOptSearch(
    space,
    max_concurrent=16,
    metric="mean_loss",
    mode="min")



scheduler = AsyncHyperBandScheduler(metric="mean_loss", mode="min")
run(objective,
    name="hyperparam_sweep1",
    search_alg=algo,
    scheduler=scheduler,
    loggers=[ray.tune.logger.CSVLogger, ray.tune.logger.JsonLogger],
    **config)
