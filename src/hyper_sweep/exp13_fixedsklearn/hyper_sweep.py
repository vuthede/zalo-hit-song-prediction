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
    'boosting_type': 'gbdt',
    'num_leaves': hp.choice('num_leaves', np.arange(45, 55, dtype=int)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0008), np.log(0.0015)),
    'reg_alpha': hp.uniform('reg_alpha', 0.075, 0.15),
    'reg_lambda': hp.uniform('reg_lambda', 0.005, 0.15),
    'seed': 99999,
    'bagging_freq': hp.choice('bagging_freq', np.arange(17, 23, dtype=int)),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.93, 0.97),
    'boost_from_average': 'false',
    'feature_fraction': hp.uniform('feature_fraction', 0.05, 0.15),
    'max_depth': -1,
    'metric': 'root_mean_squared_error',
    'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(3, 7, dtype=int)),
    'num_threads': 1,
    'tree_learner': 'serial',
    'objective': 'regression',
    'use_missing': True,
    'verbosity': 1,

}


def objective(params, reporter):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves']:
        params[parameter_name] = int(params[parameter_name])

    start = timer()
    early_stopping_rounds = 20000
    # Perform n_folds cross validation
    cv_results = perform_cv_lightgbm(df_train,
                                     chosen_features,
                                     params=params,
                                     early_stopping_rounds=early_stopping_rounds,
                                     n_folds=10)

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
    reporter(timesteps_total=1,
             mean_loss=cv_results['rmse_mean'],
             rmse_std=cv_results['rmse_std'],
             best_stopping_iter=cv_results['cv_stopping_iters'],
             best_stopping_iter_mean=cv_results['best_stopping_iter_mean'],
             best_stopping_iter_std = cv_results['best_stopping_iter_std'],
             train_time = run_time )


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
    "num_samples": 400,
    "config": {
        "iterations": 99999,  # passed to our objective --> no effect I think
    },
    "stop": {
        "timesteps_total": 99999 # Passed to our objective --> no effect I think but can be used to stop a job early I think.
    },
}

algo = HyperOptSearch(
    space,
    max_concurrent=16,
    metric="mean_loss",
    mode="min")

scheduler = AsyncHyperBandScheduler(metric="mean_loss", mode="min")
analysis = run(objective,
    name="hyperparam_sweep13",
    search_alg=algo,
    scheduler=scheduler,
    loggers=[ray.tune.logger.CSVLogger, ray.tune.logger.JsonLogger],
    **config)

import pickle
with open(r"hyper_sweep_analysis13.pickle", "wb") as output_file:
    pickle.dump(analysis, output_file)

analysis.dataframe().to_csv("results_complete13.csv")
# pip install tabulate
# tune list-trials ~/ray_results/hyperparam_sweep6/ --output result.csv
 
