from functools import partial

import torch
from hyperopt import hp
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from main import run
from src.io_utils import (
    config_to_string,
    create_paths,
    dict_to_namespace,
    namespace_to_dict,
    read_config,
)


def tune_trial(search_cfg, base_cfg=None):
    print("-->", search_cfg)
    base_dict = namespace_to_dict(base_cfg)
    base_dict.update(search_cfg)
    cfg = dict_to_namespace(base_dict)
    cfg.out_dir = tune.track.trial_dir()
    cfg.run_id = torch.randint(1000, (1,)).item()  # this actually controls the seed
    run(cfg)


def trial2string(trial):
    s = f"{trial.trial_id}"
    for k, v in trial.config.items():
        s += f"_{k}_{v}_"
    return s


def main(cmdl):
    search_name = "tune_a2c"
    base_cfg = read_config(cmdl.cfg)

    search_space = {
        "lr": hp.uniform("lr", 0.0001, 0.001),
        "nsteps": hp.choice("gamma", [10, 20, 30, 40, 50]),
    }
    hyperopt_search = HyperOptSearch(
        search_space,
        max_concurrent=8,
        reward_attr="episodic_return",
        mode="max",
    )

    analysis = tune.run(
        lambda x: tune_trial(x, base_cfg=base_cfg),
        name=search_name,
        # config=search_space,
        search_alg=hyperopt_search,
        local_dir="./results",
        num_samples=16,
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
        "--cfg", "-c", type=str, help="Path to the configuration file."
    )
    main(PARSER.parse_args())
