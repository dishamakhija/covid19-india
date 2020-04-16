import numpy as np
from hyperopt import fmin, tpe, space_eval, Trials


def hyperparam_tuning(func, search_space, max_evals, algo=tpe.suggest):
    trials = Trials()
    best=fmin(func, search_space, algo=algo, max_evals=max_evals, trials=trials)
    print("Best fit:", space_eval(search_space, best))
    trial_loss = np.asarray(trials.losses(), dtype=float)
    best_ind = np.argmin(trial_loss)
    best_loss = trial_loss[best_ind]
    print("Best Loss:", best_loss)
    return space_eval(search_space, best), trials, best_loss