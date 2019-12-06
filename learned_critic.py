""" Actor Critic, but with a twist. The Critic is learned.
"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored as clr

from liftoff import parse_opts
import rlog
from main import (
    ActionWrapper,
    TorchWrapper,
    build_agent,
    get_policy_family,
    train,
)
from src import io_utils as U


class ActorCriticEstimator(nn.Module):
    """ Model for an A2C agent.
    """

    def __init__(self, state_sz, action_space, hidden_size=64):
        super().__init__()
        self.affine1 = nn.Linear(state_sz, hidden_size)
        self.policy = get_policy_family(action_space, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        pi, value = self.policy(x), self.value(x)
        return pi, value

    def reset_policy(self):
        for m in self.policy.modules():
            try:
                print(m)
                m.reset_parameters()
            except AttributeError:
                pass

    def freeze_critic(self):
        for module_name, module in self.named_modules():
            if "policy" not in module_name and module_name != "":
                rlog.info("Freezing %s", module_name)
                module.weight.requires_grad = False
                module.bias.requires_grad = False


def run(opt):
    """ Run experiment. This function is being launched by liftoff.
    """
    U.configure_logger(opt)

    # set seed
    opt.seed = (opt.run_id + 1) * opt.base_seed
    torch.manual_seed(opt.seed)

    # configure env
    env = ActionWrapper(TorchWrapper(gym.make(opt.env_name)))
    env.seed(opt.seed)

    # build estimator
    estimator = ActorCriticEstimator(
        env.observation_space.shape[0],
        env.action_space,
        hidden_size=opt.hidden_size,
    )
    # load checkpoint and reset
    rlog.info("Loading model from %s", opt.model_state)
    estimator.load_state_dict(torch.load(opt.model_state)["policy"])
    estimator.reset_policy()
    rlog.info("Policy reset.")
    if opt.freeze_critic:
        estimator.freeze_critic()
        rlog.info("Freezed feature extractor and critic.")

    # build the agent
    policy_improvement, policy_evaluation = build_agent(
        opt, env, estimator=estimator
    )

    # log
    rlog.info(f"\n{U.config_to_string(opt)}")
    rlog.info(policy_improvement)

    # train
    try:
        train(env, policy_improvement, policy_evaluation, opt)
    except Exception as err:
        rlog.error(clr(str(err), "red", attrs=["bold"]))
        raise err


def main():
    """ Read config files using liftoff and run experiment.
    """
    opt = parse_opts()
    run(opt)


if __name__ == "__main__":
    main()