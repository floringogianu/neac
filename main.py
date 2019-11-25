""" Episodic Actor Critic
"""
from collections import namedtuple
from copy import deepcopy
import gc

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import rlog
from liftoff import parse_opts
from ray import tune
from termcolor import colored as clr

import src.io_utils as U
from src.rl_routines import Episode
from src.dnd import DND


DEVICE = torch.device("cpu")
Policy = namedtuple("Policy", ["action", "pi", "value"])
DNDPolicy = namedtuple("DNDpolicy", ["action", "pi", "value", "h"])


def train(env, policy, policy_evaluation, opt):
    """ Training routine.
    """
    log = rlog.getLogger(f"{opt.experiment}.train")
    log_fmt = (
        "[{0:6d}/{ep_cnt:5d}] R/ep={R/ep:8.2f}, V/step={V/step:8.2f}"
        + " | steps/ep={steps/ep:8.2f}, fps={fps:8.2f}."
    )
    log.reset()

    ep_cnt, step_cnt = 1, 1
    while step_cnt <= opt.training_steps:

        for state, pi, reward, state_, done in Episode(env, policy):
            policy_evaluation.learn(state, pi, reward, state_, done)
            log.put(
                reward=reward,
                value=pi.value.data.squeeze().item(),
                done=done,
                frame_no=1,
                step_no=1,
            )

            step_cnt += 1

            if step_cnt % opt.val_frequency == 0:
                validate(policy, opt, step_cnt)

        if ep_cnt % opt.log_frequency == 0:
            summary = log.summarize()
            log.info(log_fmt.format(step_cnt, **summary))
            log.trace(step=step_cnt, **summary)
            log.reset()
            gc.collect()
        ep_cnt += 1
    env.close()


def validate(policy, opt, crt_step):
    """ Validation routine """
    env = ActionWrapper(TorchWrapper(gym.make(opt.env_name)))
    policy = deepcopy(policy)
    log = rlog.getLogger(f"{opt.experiment}.valid")
    log_fmt = (
        "@{0:6d}        R/ep={R/ep:8.2f}, RunR/ep={RR/ep:8.2f}"
        + " | steps/ep={steps/ep:8.2f}, fps={fps:8.2f}."
    )
    log.reset()  # so we don't screw up the timer

    for _ in range(1, opt.val_episodes):
        with torch.no_grad():
            for _, pi, reward, _, done in Episode(env, policy):
                log.put(
                    reward=reward,
                    value=pi.value.data.squeeze().item(),
                    done=done,
                    frame_no=1,
                    step_no=1,
                )

    summary = log.summarize()
    log.info(log_fmt.format(crt_step, **summary))
    log.trace(step=crt_step, **summary)
    log.reset()
    try:
        tune.track.log(
            episodic_return=summary["R/ep"],
            running_return=summary["RR/ep"],
            value_estimate=summary["V/step"],
            train_step=crt_step,
        )
    except AttributeError as err:
        log.debug("Probably not in a ray experiment\n: %s", err)
    gc.collect()


class PolicyImprovement:
    """ Defines a behaviour.
    """

    def __init__(self, estimator):
        self.__estimator = estimator

    def act(self, state):
        """ Sample from the policy.
        """
        pi, value = self.__estimator(state)
        return Policy(pi.sample(), pi, value)

    def __call__(self, state):
        return self.act(state)

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.__estimator)


class PolicyEvaluation:
    """ Evaluates and updates the policy.
    """

    def __init__(self, policy, gamma, nsteps, optimizer, beta=0.01, **kwargs):
        self._policy = policy
        self._gamma = gamma
        self._N = nsteps
        self._beta = beta
        self._optimizer = optimizer
        self._fp32_err = 2e-07  # used to avoid division by 0
        self._policies = []  # store pi_t
        self._rewards = []  # store r_t
        self._step_cnt = 1

    def learn(self, state, policy, reward, state_, done):
        self._rewards.append(reward)
        self._policies.append(policy)

        if done or (self._step_cnt % (self._N - 1) == 0):
            self._update_policy(done, state_)

        self._step_cnt = 0 if done else self._step_cnt + 1

    def _compute_returns(self, done, state_):
        Rs = []
        R = self._policy(state_).value.detach() * (1 - done)
        for r in self._rewards[::-1]:
            R = r + self._gamma * R
            Rs.insert(0, R)
        return torch.tensor(Rs).unsqueeze(1)

    def _update_policy(self, done, state_):
        returns = self._compute_returns(done, state_).to(DEVICE)
        values = torch.cat([p.value for p in self._policies])
        log_pi = torch.cat([p.pi.log_prob(p.action) for p in self._policies])
        entropy = torch.cat([p.pi.entropy() for p in self._policies])
        advantage = returns - values

        policy_loss = (-log_pi * advantage.detach()).sum()
        critic_loss = F.smooth_l1_loss(values, returns, reduction="sum")

        self._optimizer.zero_grad()
        (policy_loss + critic_loss - self._beta * entropy.mean()).backward()
        self._optimizer.step()

        del self._rewards[:]
        del self._policies[:]


class DNDPolicyImprovement:
    """ Defines a behaviour.
    """

    def __init__(self, estimator):
        self.__estimator = estimator

    def act(self, state):
        """ Sample from the policy.
        """
        pi, value, h = self.__estimator(state)
        return DNDPolicy(pi.sample(), pi, value, h)

    def write(self, h, v, update_rule):
        self.__estimator.write(h, v, update_rule)

    def rebuild_dnd(self):
        self.__estimator.value.rebuild_tree()

    def __call__(self, state):
        return self.act(state)

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.__estimator)


class DNDPolicyEvaluation(PolicyEvaluation):
    """ Evaluates and updates the policy.
    """

    def __init__(
        self,
        policy,
        gamma,
        nsteps,
        optimizer,
        beta=0.01,
        dnd_lr=0.1,
        use_critic_grads=True,
    ):
        super().__init__(policy, gamma, nsteps, optimizer, beta=beta)
        self._dnd_lr = dnd_lr
        self._update_rule = lambda old, new: old + dnd_lr * (new - old)
        self._use_critic_grads = use_critic_grads

    def _update_policy(self, done, state_):
        returns = self._compute_returns(done, state_).to(DEVICE)

        # update policy and embedding network
        values = torch.cat([p.value for p in self._policies])
        log_pi = torch.cat([p.pi.log_prob(p.action) for p in self._policies])
        entropy = torch.cat([p.pi.entropy() for p in self._policies])
        advantage = returns - values

        policy_loss = (-log_pi * advantage.detach()).sum()
        critic_loss = F.smooth_l1_loss(values, returns, reduction="sum")
        critic_loss.data.fill_(0)

        self._optimizer.zero_grad()
        loss = policy_loss + self._beta * entropy.mean()
        if self._use_critic_grads:
            loss += critic_loss
        loss.backward()
        self._optimizer.step()

        # update DND
        for i, pi in enumerate(self._policies):
            self._policy.write(pi.h, returns.data[i], self._update_rule)

        self._rewards.clear()
        self._policies.clear()
        self._policy.rebuild_dnd()


class DiscretePolicy(nn.Module):
    def __init__(self, hidden_size, action_num):
        super().__init__()
        self._policy = nn.Linear(hidden_size, action_num)

    def forward(self, x):
        logits = self._policy(x)
        return Categorical(F.softmax(logits, dim=-1))


class ContinuousPolicy(nn.Module):
    def __init__(self, hidden_size, action_num):
        super().__init__()
        self._policy = nn.Linear(hidden_size, 2 * action_num)
        self._action_num = action_num

    def forward(self, x):
        out = self._policy(x)
        mu = torch.tanh(out[:, : self._action_num])
        std = F.softplus(out[:, self._action_num :]).clamp(1e-6, 5)
        return Normal(mu, std)


def get_policy_family(action_space, hidden_size):
    try:
        # discrete actions
        actions = action_space.n
        return DiscretePolicy(hidden_size, actions)
    except AttributeError:
        actions = action_space.shape[0]
        return ContinuousPolicy(hidden_size, actions)


class ActorCriticEstimator(nn.Module):
    def __init__(self, state_sz, action_space, hidden_size=64):
        super().__init__()
        self.affine1 = nn.Linear(state_sz, hidden_size)
        self.policy = get_policy_family(action_space, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        pi, value = self.policy(x), self.value(x)
        return pi, value


class DNDEstimator(nn.Module):
    def __init__(self, state_sz, action_space, dnd_size=20_000, hidden_size=64):
        super().__init__()
        self.affine1 = nn.Linear(state_sz, hidden_size)
        self.policy = get_policy_family(action_space, hidden_size)
        self.value = DND(hidden_size, torch.device("cpu"), max_size=dnd_size)

    def forward(self, x):
        h = F.relu(self.affine1(x))
        if self.value.ready:
            return self.policy(h), self.value.lookup(h), h
        self.value.write(h, torch.tensor([[0.0]]), lambda v, v_: v_)
        return self.policy(h), torch.tensor([[0.0]]), h

    def write(self, h, v, update_rule):
        self.value.write(h, v, update_rule)


class TorchWrapper(gym.ObservationWrapper):
    """ Applies a couple of transformations depending on the mode.
        Receives numpy arrays and returns torch tensors.
    """

    def __init__(self, env, device=DEVICE):
        super().__init__(env)
        self._device = device

    def observation(self, obs):
        return torch.from_numpy(obs).float().unsqueeze(0).to(self._device)


class ActionWrapper(gym.ActionWrapper):
    """ Torch to Gym-compatible actions.
    """

    def __init__(self, env):
        super().__init__(env)
        self._action_type = "Z" if hasattr(env, "n") else "R"

    def action(self, action):
        if self._action_type == "Z":
            return action.cpu().item()
        return action.squeeze().cpu().numpy()


AGENTS = {
    "a2c": {
        "estimator": ActorCriticEstimator,
        "policy_improvement": PolicyImprovement,
        "policy_evaluation": PolicyEvaluation,
    },
    "neac": {
        "estimator": DNDEstimator,
        "policy_improvement": DNDPolicyImprovement,
        "policy_evaluation": DNDPolicyEvaluation,
    },
}


def build_agent(opt, env):
    if opt.algo == "a2c":
        kw = {"hidden_size": opt.hidden_size}
    elif opt.algo == "neac":
        kw = {"hidden_size": opt.dnd.key_size, "dnd_size": opt.dnd.size}
    else:
        raise ValueError(f"{opt.algo} is not a known option.")

    estimator = AGENTS[opt.algo]["estimator"](
        env.observation_space.shape[0], env.action_space, **kw
    ).to(DEVICE)

    policy = AGENTS[opt.algo]["policy_improvement"](estimator)
    policy_evaluation = AGENTS[opt.algo]["policy_evaluation"](
        policy,
        opt.gamma,
        opt.nsteps,
        optim.Adam(estimator.parameters(), lr=opt.lr, eps=1e-05),
        beta=opt.beta_entropy,
        dnd_lr=opt.dnd.lr if hasattr(opt, "dnd") else None,
        use_critic_grads=opt.dnd.use_critic_grads if hasattr(opt, "dnd") else True
    )
    return policy, policy_evaluation


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

    # build the agent
    policy_improvement, policy_evaluation = build_agent(opt, env)

    # log
    rlog.info(f"\n{U.config_to_string(opt)}")
    rlog.info(policy_improvement)

    # train
    try:
        train(env, policy_improvement, policy_evaluation, opt)
    except Exception as err:
        rlog.error(clr(err, "red", attrs=["bold"]))
        raise err


def main():
    """ Read config files using liftoff and run experiment.
    """
    opt = parse_opts()
    run(opt)


if __name__ == "__main__":
    main()
