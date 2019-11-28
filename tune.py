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
    config_to_string,
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
        if isinstance(v, float):
            v = f"{v:.4f}"[:6]
        s += f"_{k}_{v}_"
    return s


def get_search_space(search_cfg):
    good_inits = None
    search_space = {}
    if "hyperopt" in search_cfg:
        flat_cfg = flatten_dict(search_cfg["hyperopt"])
        for k, v in flat_cfg.items():
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
    if "good_inits" in search_cfg:
        good_inits = [flatten_dict(d) for d in search_cfg["good_inits"]]
    return good_inits, search_space


def main(cmdl):
    base_cfg = namespace_to_dict(read_config(Path(cmdl.cfg) / "default.yaml"))
    search_cfg = namespace_to_dict(read_config(Path(cmdl.cfg) / "search.yaml"))

    print(config_to_string(cmdl))
    print(config_to_string(dict_to_namespace(search_cfg)))

    # the search space
    good_init, search_space = get_search_space(search_cfg)

    search_name = "{timestep}_tune_{experiment_name}{dev}".format(
        timestep="{:%Y%b%d-%H%M%S}".format(datetime.now()),
        experiment_name=base_cfg["experiment"],
        dev="_dev" if cmdl.dev else "",
    )

    # search algorithm
    hyperopt_search = HyperOptSearch(
        search_space,
        metric="episodic_return",
        mode="max",
        max_concurrent=cmdl.workers,
        points_to_evaluate=good_init,
    )

    # early stopping
    scheduler = ASHAScheduler(
        time_attr="train_step",
        metric="episodic_return",
        mode="max",
        max_t=base_cfg["training_steps"],  # max length of the experiment
        grace_period=cmdl.grace_steps,  # stops after 20 logged steps
        brackets=3,  # don't know what this does
    )

    analysis = tune.run(
        lambda x: tune_trial(x, base_cfg=base_cfg),
        name=search_name,
        # config=search_space,
        search_alg=hyperopt_search,
        scheduler=scheduler,
        local_dir="./results",
        num_samples=cmdl.trials,
        trial_name_creator=trial2string,
    )

    dfs = analysis.trial_dataframes
    for i, (key, df) in enumerate(dfs.items()):
        print("saving: ", key)
        df.to_pickle(f"{key}/trial_df.pkl")


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(description="NeuralEpisodicActorCritic")
    PARSER.add_argument(
        "--cfg", "-c", type=str, help="Path to the configuration folder."
    )
    PARSER.add_argument(
        "--trials", "-t", default=16, type=int, help="Number of total trials."
    )
    PARSER.add_argument(
        "--workers",
        "-w",
        default=8,
        type=int,
        help="Number of available processes.",
    )
    PARSER.add_argument(
        "--grace-steps",
        "-g",
        default=1_000,
        type=int,
        help="Grace period in env steps.",
    )
    PARSER.add_argument("--dev", "-d", action="store_true", help="Dev mode.")
    main(PARSER.parse_args())
