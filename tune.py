""" Using Ray Tune to find hyperparams.
"""
from datetime import datetime
from pathlib import Path

import ray
import torch
import yaml
from hyperopt import hp  # pylint: disable=unused-import
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from main import configure_experiment, learn, validate
from src.io_utils import (
    config_to_string,
    dict_to_namespace,
    expand_dict,
    flatten_dict,
    namespace_to_dict,
    read_config,
    recursive_update,
)
from src.rl_routines import train_rounds


@ray.remote(num_cpus=1)
class Seed:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.cfg.seed = seed

        # create paths
        self.cfg.out_dir = f"{cfg.out_dir}/{seed}"
        Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)

        self.env, self.pi, self.pi_evaluation = configure_experiment(self.cfg)

    def train(self, train_round):
        env, policy, policy_evaluation = self.env, self.pi, self.pi_evaluation
        _, end = train_round

        # learn for a number of steps
        learn(env, policy, policy_evaluation, train_round)

        # validate
        results = validate(policy, self.cfg, end)

        # save the agent
        if hasattr(self.cfg, "save_agent") and self.cfg.save_agent:
            torch.save(
                {
                    "step": end,
                    "policy": policy.estimator_state(),
                    "R/ep": results["R_ep"],
                },
                f"{self.cfg.out_dir}/policy_step_{end:07d}.pth",
            )
        return results


def normalize(xs, rescaled=True):
    """ Normalize and rescale """
    # TODO: think again about this
    val = xs.mean().pow(2) - xs.var()
    if rescaled:
        rmax = 1_000_000
        rmin = -222_222.2188
        return (val - rmin) / (rmax - rmin)
    return val


def tune_trial(search_cfg, base_cfg=None, get_objective=None):
    """ Update the base config with the search config returned by `tune`,
        convert to a Namespace and run a trial.
    """
    cfg = recursive_update(base_cfg, expand_dict(search_cfg))

    cfg["out_dir"] = tune.track.trial_dir()  # add output dir for saving stuff
    with open(f"{cfg['out_dir']}/cfg.yaml", "w") as file:
        yaml.safe_dump(cfg, file, default_flow_style=False)  # save the new cfg

    cfg = dict_to_namespace(cfg)

    # Start three different Seeds on separate processes
    seeds = [
        Seed.remote(cfg, seed.item()) for seed in torch.randint(1000, (3,))
    ]

    for start, end in train_rounds(cfg.training_steps, cfg.val_frequency):
        # launch training and validation tasks
        tasks = [seed.train.remote((start, end)) for seed in seeds]
        # and imediately collect them
        results = [ray.get(task) for task in tasks]

        # we log the mean of three seeds hoping that this way
        # tune will find more robust hyperparams.
        episodic_returns = torch.tensor([r["R_ep"] for r in results])
        objective = mean_returns = torch.mean(episodic_returns)
        if get_objective is not None:
            # or we rescale the mean with the variance of the seeds
            # so that we discourage high variance configs.
            objective = get_objective(episodic_returns)
        tune.track.log(
            criterion=objective.item(),
            episodic_return=mean_returns.item(),
            train_step=end,
        )


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
        metric="criterion",
        mode="max",
        max_concurrent=cmdl.workers,
        points_to_evaluate=good_init,
    )

    # early stopping
    scheduler = ASHAScheduler(
        time_attr="train_step",
        metric="criterion",
        mode="max",
        max_t=base_cfg["training_steps"],  # max length of the experiment
        grace_period=cmdl.grace_steps,  # stops after 20 logged steps
        brackets=3,  # don't know what this does
    )

    analysis = tune.run(
        lambda x: tune_trial(x, base_cfg=base_cfg, get_objective=None),
        name=search_name,
        # config=search_space,
        search_alg=hyperopt_search,
        scheduler=scheduler,
        local_dir="./results",
        num_samples=cmdl.trials,
        trial_name_creator=trial2string,
        resources_per_trial={"cpu": 3},
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
