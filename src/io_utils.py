""" I/O utils.
"""
import os
import subprocess
from argparse import Namespace
from datetime import datetime

import yaml
from termcolor import colored as clr

import rlog


def configure_logger(opt):
    """ Configures the logger.
    """
    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True)
    train_log = rlog.getLogger(opt.experiment + ".train")
    train_log.addMetrics(
        rlog.AvgMetric("R_ep", metargs=["reward", "done"]),
        rlog.AvgMetric("V_step", metargs=["value", 1]),
        rlog.AvgMetric("v_mse_loss", metargs=["v_mse", 1]),
        rlog.AvgMetric("v_hub_loss", metargs=["v_hub", 1]),
        rlog.SumMetric("ep_cnt", resetable=False, metargs=["done"]),
        rlog.AvgMetric("steps_ep", metargs=["step_no", "done"]),
        rlog.FPSMetric("fps", metargs=["frame_no"]),
    )
    train_log.log_fmt = (
        "[{0:6d}/{ep_cnt:5d}] R/ep={R_ep:8.2f}, V/step={V_step:8.2f}"
        + " | steps/ep={steps_ep:8.2f}, fps={fps:8.2f}."
    )
    val_log = rlog.getLogger(opt.experiment + ".valid")
    val_log.addMetrics(
        rlog.AvgMetric("R_ep", metargs=["reward", "done"]),
        rlog.AvgMetric(
            "RR_ep", resetable=False, eps=0.8, metargs=["reward", "done"]
        ),
        rlog.AvgMetric("V_step", metargs=["value", 1]),
        rlog.AvgMetric("steps_ep", metargs=["frame_no", "done"]),
        rlog.FPSMetric("fps", metargs=["frame_no"]),
    )
    if hasattr(opt.log, "detailed") and opt.log.detailed:
        val_log.addMetrics(
            rlog.ValueMetric("Vhist", metargs=["value"], tb_type="histogram")
        )
    val_log.log_fmt = (
        "@{0:6d}        R/ep={R_ep:8.2f}, RunR/ep={RR_ep:8.2f}"
        + " | steps/ep={steps_ep:8.2f}, fps={fps:8.2f}."
    )


def config_to_string(
    cfg: Namespace, indent: int = 0, color: bool = True
) -> str:
    """Creates a multi-line string with the contents of @cfg."""

    text = ""
    for key, value in cfg.__dict__.items():
        ckey = clr(key, "yellow", attrs=["bold"]) if color else key
        text += " " * indent + ckey + ": "
        if isinstance(value, Namespace):
            text += "\n" + config_to_string(value, indent + 2, color=color)
        else:
            cvalue = clr(str(value), "white") if color else str(value)
            text += cvalue + "\n"
    return text


class YamlNamespace(Namespace):
    """ PyLint will trigger `no-member` errors for Namespaces constructed
    from yaml files. I am using this inherited class to target an
    `ignored-class` rule in `.pylintrc`.
    """


def create_paths(args: Namespace) -> Namespace:
    """ Creates directories for containing experiment results.
    """
    time_stamp = "{:%Y%b%d-%H%M%S}".format(datetime.now())
    if not hasattr(args, "out_dir") or args.out_dir is None:
        if not os.path.isdir("./results"):
            os.mkdir("./results")
        out_dir = f"./results/{time_stamp}_{args.experiment:s}"
        os.mkdir(out_dir)
        args.out_dir = out_dir
    elif not os.path.isdir(args.out_dir):
        raise Exception(f"Directory {args.out_dir} does not exist.")

    if not hasattr(args, "run_id"):
        args.run_id = 0

    return args


def dict_to_namespace(dct: dict) -> Namespace:
    """Deep (recursive) transform from Namespace to dict"""
    namespace = YamlNamespace()
    for key, value in dct.items():
        name = key.rstrip("_")
        if isinstance(value, dict) and not key.endswith("_"):
            setattr(namespace, name, dict_to_namespace(value))
        else:
            setattr(namespace, name, value)
    return namespace


def namespace_to_dict(namespace: Namespace) -> dict:
    """Deep (recursive) transform from Namespace to dict"""
    dct: dict = {}
    for key, value in namespace.__dict__.items():
        if isinstance(value, Namespace):
            dct[key] = namespace_to_dict(value)
        else:
            dct[key] = value
    return dct


def flatten_dict(dct: dict, prev_key: str = None) -> dict:
    """Recursive flattening a dict"""
    flat_dct: dict = {}
    for key, value in dct.items():
        new_key = f"{prev_key}.{key}" if prev_key is not None else key
        if isinstance(value, dict):
            flat_dct.update(flatten_dict(value, prev_key=new_key))
        else:
            flat_dct[new_key] = value
    return flat_dct


def recursive_update(d: dict, u: dict) -> dict:
    "Recursively update `d` with stuff in `u`."
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _expand_from_keys(keys: list, value: object) -> dict:
    """ Expand [a, b c] to {a: {b: {c: value}}} """
    dct = d = {}
    while keys:
        key = keys.pop(0)
        d[key] = {} if keys else value
        d = d[key]
    return dct


def expand_dict(flat_dict: dict) -> dict:
    """ Expand {a: va, b.c: vbc, b.d: vbd} to {a: va, b: {c: vbc, d: vbd}}.
        If not clear from above we want:
        {'lr':              0.0011,
         'gamma':           0.95,
         'dnd.size':        2000,
         'dnd.lr':          0.77,
         'dnd.sched.end':   0.0,
         'dnd.sched.steps': 1000
        }
        to this:
        {'lr': 0.0011,
         'gamma': 0.95,
         'dnd': {'size': 2000,
                 'lr': 0.77,
                 'sched': {'end': 0.0,
                           'steps': 1000
        }}}
    """
    exp_dict = {}
    for key, value in flat_dict.items():
        if "." in key:
            keys = key.split(".")
            key_ = keys.pop(0)
            if key_ not in exp_dict:
                exp_dict[key_] = _expand_from_keys(keys, value)
            else:
                exp_dict[key_] = recursive_update(
                    exp_dict[key_], _expand_from_keys(keys, value)
                )
        else:
            exp_dict[key] = value
    return exp_dict


def read_config(cfg_path):
    """ Read a config file and return a namespace.
    """
    with open(cfg_path) as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)
    return dict_to_namespace(config_data)


def get_git_info() -> str:
    """ Return sha@branch.
    This can maybe be used when restarting experiments. We can trgger a
    warning if the current code-base does not match the one we are trying
    to resume from.
    """
    cmds = [
        ["git", "rev-parse", "--short", "HEAD"],  # short commit sha
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # branch name
    ]
    res = []
    for cmd in cmds:
        res.append(subprocess.check_output(cmd).strip().decode("utf-8"))
    return "@".join(res)
