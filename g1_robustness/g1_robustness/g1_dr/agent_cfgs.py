"""PPO runner configs: identical to Isaac Lab's stock G1 flat config (same network, hyper-parameters, budget).
Only ``experiment_name`` differs so the logs of the two groups land in separate folders."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1FlatBaselinePPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_flat_baseline"


@configclass
class G1FlatDRPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_flat_dr"
