""" Evaluate the a greedy policy based on the learned V(s) of each saved
model.
"""
import argparse
from copy import deepcopy
from collections import defaultdict
from functools import partial
from pathlib import Path

import gym
import torch
import torch.nn.functional as F
import rlog

from main import ActionWrapper, ActorCriticEstimator, TorchWrapper, build_agent
from src.io_utils import config_to_string, read_config
from src.dnd import _hash, _get_achlioptas


def greedy_pi(env, policy, gamma):
    """ Compute V(s) = r + gamma * V(s') for all a and choose the one that
        maximizes V(s).
    """
    value_estimates = []

    for action in range(env.action_space.n):
        _env = deepcopy(env)  # copy the env
        state_, reward, done, _ = _env.step(action)  # take a step
        if not done:
            v_ = policy.act(state_).value  # get V(s')
            value_estimates.append(reward + gamma * v_)  # compute V(s)
        else:
            value_estimates.append(reward)
    return value_estimates.index(max(value_estimates))  # return max_a V(s)


def greedy_validation(env, policy, gamma, val_episodes=100):
    """ Validate the greedy policy based on V(s).
    """

    returns = 0
    for ep_no in range(1, val_episodes):
        state, done, R = env.reset(), False, 0

        while not done:
            with torch.no_grad():
                # action = policy.act(state).action
                action = greedy_pi(env, policy, gamma)
                state, reward, done, _ = env.step(action)
                R += reward

        returns += R
    return returns / ep_no


def offline_validation(crt_step, policy, dset, opt):
    log = rlog.getLogger(opt.experiment + ".off_valid")
    log.info(f"Validating chkp@{crt_step:08d} on offline data...")

    # Iterate through the data
    for step, (state, stats) in enumerate(dset["dset"].items()):
        with torch.no_grad():
            pi = policy(state)
        loss = F.mse_loss(pi.value, torch.tensor([[stats["Gt"]]]))

        log.put(value=pi.value.squeeze().item(), off_mse=loss.squeeze().item())

        if step % 100_000 == 0 and step != 0:
            fstr = "partial@ {0:8d} V/step={V_step:8.2f}, off_mse/ep={off_mse:8.2f}"
            log.info(fstr.format(step, **log.summarize()))

    fstr = "Full@ {0:8d} V/step={V_step:8.2f}, off_mse/ep={off_mse:8.2f}"
    summary = log.summarize()
    log.info(fstr.format(crt_step, **summary))
    log.trace(step=crt_step, **summary)
    log.reset()


def configure_eval(cmdl, opt, path):
    """ Builds the objects required for conducting the experiment, env,
        estimator, policy, etc.
    """
    # build env
    env = ActionWrapper(TorchWrapper(gym.make(opt.env_name)))
    env.seed(cmdl.seed)

    # build estimator
    estimator = ActorCriticEstimator(
        env.observation_space.shape[0],
        env.action_space,
        hidden_size=opt.hidden_size,
    )
    # load checkpoint
    estimator.load_state_dict(torch.load(path)["policy"])
    # build the agent
    policy_improvement, _ = build_agent(opt, env, estimator=estimator)

    return env, policy_improvement


def _compute_returns(rewards, gamma):
    R, Rs = 0, []
    for r in rewards[::-1]:
        R = r + gamma * R
        Rs.insert(0, R)
    return Rs


def build_validation_dset(env, policy, gamma, hash_fn, val_episodes=5_000):
    """ Uses a policy to sample an environment and estimate V(s_t).
    """
    state_returns, steps = defaultdict(lambda: {"state": None, "Rs": []}), 0

    tot_returns = []
    for ep_no in range(1, val_episodes + 1):
        states, rewards = [], []
        state, done = env.reset(), False

        ep_return = 0
        while not done:
            with torch.no_grad():

                states.append(state.clone())
                pi = policy.act(state)
                state, reward, done, _ = env.step(pi.action)
                rewards.append(reward)

                steps += 1
                ep_return += reward

        # append the returns
        returns = _compute_returns(rewards, gamma)

        for S, R in zip(states, returns):
            key = hash_fn(S)
            state_returns[key]["state"] = state.clone()
            state_returns[key]["Rs"].append(R)

        tot_returns.append(ep_return)

        if ep_no % 100 == 0:
            print(
                "[{:5d}] states: {:6d} steps: {:8d} | {:3.1f}% unique. R/ep={:3.1f}.".format(
                    ep_no,
                    len(state_returns),
                    steps,
                    len(state_returns) / steps * 100,
                    torch.tensor(tot_returns).mean(),
                )
            )
            tot_returns = []

    val_dset = {}
    for state_hash, payload in state_returns.items():
        Rs = torch.tensor(payload["Rs"])

        val_dset[payload["state"]] = {
            "Gt": Rs.mean().item(),
            "Gt_std": Rs.std().item(),
            "Rs": Rs,
            "key": state_hash,
        }

    return {
        "meta": {
            "states": len(state_returns),
            "step_no": steps,
            "ep_no": ep_no,
            "unique": len(state_returns) / steps * 100,
        },
        "dset": val_dset,
    }


def main(cmdl):
    """ Entry point.
    """
    opt = read_config(Path(cmdl.experiment_path) / "cfg.yaml")
    chkpt_paths = sorted(
        Path(cmdl.experiment_path).glob("*.pth"),
        key=lambda path: int(path.stem.split("_")[2]),
    )
    chkpt_paths = [(int(path.stem.split("_")[2]), path) for path in chkpt_paths]

    print(config_to_string(cmdl))
    print(config_to_string(opt))

    if cmdl.build_val_dset:
        perf = [(torch.load(path)["R/ep"], path) for _, path in chkpt_paths]
        best_score, path = max(perf, key=lambda x: x[0])
        print(f"Loading {path} with total return {best_score}.")
        env, policy = configure_eval(cmdl, opt, path)
        achlioptas = _get_achlioptas(8, 4)
        val_dset = build_validation_dset(
            env,
            policy,
            opt.gamma,
            partial(_hash, decimals=cmdl.decimals, rnd_proj=achlioptas),
        )

        val_dset["meta"]["agent"] = path
        val_dset["meta"]["decimals"] = cmdl.decimals
        val_dset["meta"]["rnd_proj"] = achlioptas
        for k, v in val_dset["meta"].items():
            print(f"{k:12}", v)
        torch.save(val_dset, f"./val_dsets/{env.spec.id}.pkl")
    elif cmdl.offline_validation:
        rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True)
        log = rlog.getLogger(opt.experiment + ".off_valid")
        log.addMetrics(
            [
                rlog.AvgMetric("V_step", metargs=["value", 1]),
                rlog.AvgMetric("off_mse", metargs=["off_mse", 1]),
            ]
        )
        log.info("Loading dataset...")
        dset = torch.load(f"./val_dsets/{opt.env_name}.pkl")
        for step, path in chkpt_paths:
            env, policy = configure_eval(cmdl, opt, path)
            offline_validation(step, policy, dset, opt)
    else:
        for step, path in chkpt_paths:
            env, policy = configure_eval(cmdl, opt, path)
            avg_return = greedy_validation(env, policy, opt.gamma)
            print("[{0:8d}]   R/ep={1:8.2f}.".format(step, avg_return))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="V(S) greedy eval.")
    PARSER.add_argument(
        "experiment_path",
        type=str,
        help="Path to the experiment containing saved models.",
    )
    PARSER.add_argument(
        "--seed", "-s", type=int, default=42, help="Env seed, defaults to 42."
    )
    PARSER.add_argument(
        "--decimals",
        "-d",
        type=int,
        default=2,
        help="No of decimal places to round the states or the projected states to. Defaults to 2.",
    )
    PARSER.add_argument(
        "--build-val-dset",
        action="store_true",
        help="Samples an environment and stores state-action value returns.",
    )
    PARSER.add_argument(
        "--offline-validation", action="store_true", help="TODO."
    )
    main(PARSER.parse_args())
