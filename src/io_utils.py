""" I/O utils.
"""
from argparse import Namespace
import rlog
from termcolor import colored as clr


def configure_logger(opt):
    """ Configures the logger.
    """
    rlog.init(opt.experiment, path=opt.out_dir)
    train_log = rlog.getLogger(opt.experiment + ".train")
    train_log.addMetrics(
        [
            rlog.AvgMetric("R/ep", metargs=["reward", "done"]),
            rlog.AvgMetric("V/step", metargs=["value", 1]),
            rlog.SumMetric("ep_cnt", resetable=False, metargs=["done"]),
            rlog.AvgMetric("steps/ep", metargs=["step_no", "done"]),
            rlog.FPSMetric("learning_fps", metargs=["frame_no"]),
        ]
    )
    test_log = rlog.getLogger(opt.experiment + ".test")
    test_log.addMetrics(
        [
            rlog.AvgMetric("R/ep", metargs=["reward", "done"]),
            rlog.SumMetric("ep_cnt", resetable=False, metargs=["done"]),
            rlog.AvgMetric("steps/ep", metargs=["frame_no", "done"]),
            rlog.FPSMetric("test_fps", metargs=["frame_no"]),
            rlog.MaxMetric("max_q", metargs=["qval"]),
        ]
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
