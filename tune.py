""" Using Ray Tune to find hyperparams.
"""
from datetime import datetime
from pathlib import Path

import torch
from hyperopt import hp
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from main import run
from src.io_utils import (
    dict_to_namespace,
    namespace_to_dict,
    read_config,
    flatten_dict,
    expand_dict,
)


def tune_trial(search_cfg, base_cfg=None):
    print("-->", search_cfg)
    base_cfg.update(expand_dict(search_cfg))
    cfg = dict_to_namespace(base_cfg)
    cfg.out_dir = tune.track.trial_dir()
    cfg.run_id = torch.randint(1000, (1,)).item()
    run(cfg)


def trial2string(trial):
    s = f"{trial.trial_id}"
    for k, v in trial.config.items():
        s += f"_{k}_{v}_"
    return s


def get_search_space(search_cfg):

    search_space = {}
    if "hyperopt" in search_cfg:
        search_cfg = flatten_dict(search_cfg["hyperopt"])
        for k, v in search_cfg.items():
            if v[0] == "uniform":
                args = [k, *v[1]]
            elif v[0] == "choice":
                args = [k, v[1]]
            else:
                raise ValueError("Unknown HyperOpt space: ", v[0])
            try:
                search_space[k] = eval(f"hp.{v[0]}")(*args)
            except Exception as err:
                print(k, v, args)
                raise err

    else:
        raise ValueError("Unknown search space.", search_cfg)
    return search_space


def main(cmdl):
    max_workers = 32
    trials = 512  ## whoa!

    base_cfg = namespace_to_dict(read_config(Path(cmdl.cfg) / "default.yaml"))
    search_cfg = namespace_to_dict(read_config(Path(cmdl.cfg) / "search.yaml"))

    # the search space
    search_space = get_search_space(search_cfg)

    search_name = "{timestep}_tune_{algo_name}".format(
        timestep="{:%Y%b%d-%H%M%S}".format(datetime.now()),
        algo_name=base_cfg["algo"],
    )

    # search algorithm
    hyperopt_search = HyperOptSearch(
        search_space,
        metric="episodic_return",
        mode="max",
        max_concurrent=max_workers,
    )

    # early stopping
    scheduler = ASHAScheduler(
        metric="episodic_return",
        mode="max",
        max_t=200,  # max length of the experiment
        grace_period=50,  # stops after 20 logged steps
        brackets=3,  # don't know what this does
    )

    analysis = tune.run(
        lambda x: tune_trial(x, base_cfg=base_cfg),
        name=search_name,
        # config=search_space,
        search_alg=hyperopt_search,
        scheduler=scheduler,
        local_dir="./results",
        num_samples=trials,
        trial_name_creator=trial2string,
    )

    dfs = analysis.trial_dataframes
    for i, (key, df) in enumerate(dfs.items()):
        print("saving: ", key)
        df.to_pickle(f"./results/{search_name}/trial_{i}_df.pkl")


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(description="NeuralEpisodicActorCritic")
    PARSER.add_argument(
        "--cfg", "-c", type=str, help="Path to the configuration folder."
    )
    main(PARSER.parse_args())
