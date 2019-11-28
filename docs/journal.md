# Project log


## Nov 26, robustness check
---

Select the best looking hyperparameters found by `ray.tune` for each
environment and train on ten seeds.


| Env           |      A2C               |         neAC           |
|---------------|:----------------------:|:----------------------:|
| LunarLander-C | `b5cf56c4`, `6ead0806` | `4dc127e8`, `97306f06` |
| LunarLander-D |                        |                        |
| Acrobot       | `af32a06a`, `7d018958` | `aa315a12`, `1a3ed1d8` |
| BipedalWlaker |                        |                        |

Tune results folders:

```bash
2019Nov23-111409_tune_a2c_dev       <-- acrobot
2019Nov23-113341_tune_neac_dev      <-- acrobot
2019Nov23-195637_tune_a2c_walker
2019Nov23-195911_tune_neac_walker
2019Nov24-191525_tune_neac_lld
2019Nov24-192502_tune_neac_walker
2019Nov25-124623_tune_neac_llc      <-- lunar-lander-c
2019Nov26-114401_tune_a2c_llc       <-- lunar-lander-c
```

I only managed to start `a2c_confirm` with the four experiments above.


## Nov 27, confirm neAC too 
---

- [x] Configure and launch **neAC** experiments too.
- [ ] Explore and extend last night's **A2C** experiments.
- [ ] Rerun `ray.tune` on Acrobot.
- [ ] Decide on what's next.

Initial results look bad on Acrobot, these hyperparameters are not robust
troughout the training. I am adding two more configs which appear to be
more stable.

| Env           |      A2C               |         neAC           |
|---------------|:----------------------:|:----------------------:|
| LunarLander-C | `b5`, `6e`, `6912ec4c` | `4dc127e8`, `97306f06` |
| Acrobot       | `af`, `7d`, `7c1a7aae` | `aa315a12`, `1a3ed1d8` |

**At this point I think I should run `ray.tune` again on Acrobot-v1**. The
models don't seem trained or robust.

Results with the two additional configs are dubious:

![a2c confirm on the two envs above](./img/27_nov_a2c_confirm_three.png)

These two additional configs, `6912ec4c`, `7c1a7aae` have been chosen for
their stability. However they are not showing it.

I am adding some more:

| Env           |      A2C                  |         neAC           |
|---------------|:-------------------------:|:----------------------:|
| LunarLander-C | `b5`,`6e`,`69`,`9e7953d8` | `4dc127e8`, `97306f06` |


It's becoming clear that `ray.tune` is simply selecting some lucky seeds. All four configurations below should have solved `LunarLander`.

![a2c confirm on the two envs above](./img/27_nov_a2c_confirm_four.png)

### Summary

- Check the issue of low performance configurations found by `ray.tune` on the
discrete case too.
- Implement and look what happens with an optimal Value function.
- Look at gradients.