""" Episodic Actor Critic
"""
from collections import namedtuple
import gc

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import rlog
from liftoff import parse_opts

import src.io_utils as U
from src.rl_routines import Episode
from src.dnd import DND


DEVICE = torch.device("cpu")
Policy = namedtuple("Policy", ["action", "pi", "value"])
DNDPolicy = namedtuple("DNDpolicy", ["action", "pi", "value", "h"])


class PolicyImprovement:
    """ Defines a behaviour.
    """

    def __init__(self, estimator):
        self.__estimator = estimator

    def act(self, state):
        """ Sample from the policy.
        """
        logits, value = self.__estimator(state)
        pi = Categorical(F.softmax(logits, dim=-1))
        return Policy(pi.sample(), pi, value)

    def __call__(self, state):
        return self.act(state)


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
        return torch.tensor(Rs)

    def _update_policy(self, done, state_):
        returns = self._compute_returns(done, state_).to(DEVICE)
        values = torch.cat([p.value for p in self._policies]).squeeze(1)
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
        logits, value, h = self.__estimator(state)
        pi = Categorical(F.softmax(logits, dim=-1))
        return DNDPolicy(pi.sample(), pi, value, h)

    def write(self, h, v, update_rule):
        self.__estimator.write(h, v, update_rule)

    def rebuild_dnd(self):
        self.__estimator.value.rebuild_tree()

    def __call__(self, state):
        return self.act(state)


class DNDPolicyEvaluation(PolicyEvaluation):
    """ Evaluates and updates the policy.
    """

    def __init__(self, policy, gamma, nsteps, optimizer, beta=0.01, dnd_lr=0.1):
        super().__init__(policy, gamma, nsteps, optimizer, beta=beta)
        self._dnd_lr = dnd_lr
        self._update_rule = lambda old, new: old + dnd_lr * (new - old)

    def _update_policy(self, done, state_):
        returns = self._compute_returns(done, state_).to(DEVICE)

        # update DND
        for i, pi in enumerate(self._policies):
            self._policy.write(pi.h, returns.data[i], self._update_rule)

        # update policy and embedding network
        values = torch.cat([p.value for p in self._policies]).squeeze(1)
        log_pi = torch.cat([p.pi.log_prob(p.action) for p in self._policies])
        entropy = torch.cat([p.pi.entropy() for p in self._policies])
        advantage = returns - values

        policy_loss = (-log_pi * advantage.detach()).sum()
        critic_loss = F.smooth_l1_loss(values, returns, reduction="sum")

        self._optimizer.zero_grad()
        (policy_loss + critic_loss - self._beta * entropy.mean()).backward()
        self._optimizer.step()

        self._rewards.clear()
        self._policies.clear()
        self._policy.rebuild_dnd()


class ActorCriticEstimator(nn.Module):
    def __init__(self, state_sz, action_num, hidden_size=64):
        super().__init__()
        self.affine1 = nn.Linear(state_sz, hidden_size)
        self.policy = nn.Linear(hidden_size, action_num)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.policy(x), self.value(x)


class DNDEstimator(nn.Module):
    def __init__(self, state_sz, action_num, dnd_size=20_000, hidden_size=64):
        super().__init__()
        self.affine1 = nn.Linear(state_sz, hidden_size)
        self.policy = nn.Linear(hidden_size, action_num)
        self.value = DND(hidden_size, torch.device("cpu"), max_size=dnd_size)

    def forward(self, x):
        h = F.relu(self.affine1(x))
        if self.value.ready:
            return self.policy(h), self.value.lookup(h), h
        self.value.write(h, torch.tensor([[0.0]]), lambda v, v_: v_)
        return self.policy(h), torch.tensor([[0.0]]), h

    def write(self, h, v, update_rule):
        self.value.write(h, v, update_rule)


def train(env, policy, policy_evaluation, opt):
    """ Training routine.
    """
    log = rlog.getLogger(f"{opt.experiment}.train")
    log_fmt = (
        "[{0:6d}/{ep_cnt:5d}] R/ep={R/ep:8.2f}, V/step={V/step:8.2f}"
        + " | steps/ep={steps/ep:8.2f}, fps={learning_fps:8.2f}."
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

        if ep_cnt % opt.log_interval == 0:
            summary = log.summarize()
            log.info(log_fmt.format(step_cnt, **summary))
            log.trace(step=step_cnt, **summary)
            log.reset()
            gc.collect()
        ep_cnt += 1
    env.close()


class TorchWrapper(gym.ObservationWrapper):
    """ Applies a couple of transformations depending on the mode.
        Receives numpy arrays and returns torch tensors.
    """

    def __init__(self, env, device=DEVICE):
        super().__init__(env)
        self._device = device

    def observation(self, obs):
        return torch.from_numpy(obs).float().unsqueeze(0).to(self._device)


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
        env.observation_space.shape[0], env.action_space.n, **kw
    ).to(DEVICE)
    policy = AGENTS[opt.algo]["policy_improvement"](estimator)
    policy_evaluation = AGENTS[opt.algo]["policy_evaluation"](
        policy,
        opt.gamma,
        opt.nsteps,
        optim.Adam(estimator.parameters(), lr=opt.lr, eps=1e-05),
        beta=opt.beta_entropy,
        dnd_lr=opt.dnd.lr if hasattr(opt, "dnd") else None,
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
    env = TorchWrapper(gym.make(opt.env_name))
    env.seed(opt.seed)

    rlog.info(f"\n{U.config_to_string(opt)}")

    # build the agent
    policy, policy_evaluation = build_agent(opt, env)

    # train
    train(env, policy, policy_evaluation, opt)


def main():
    """ Read config files using liftoff and run experiment.
    """
    opt = parse_opts()
    run(opt)


if __name__ == "__main__":
    main()
