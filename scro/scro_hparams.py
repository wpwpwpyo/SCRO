from dataclasses import dataclass
from math import isfinite
from typing import List, Literal

from util.hparams import HyperParams


@dataclass
class SCROHyperParams(HyperParams):
    """Compact SCRO configuration surface.

    Stage1 always uses a cold start, the full numerical-rank spectrum, no
    per-sample clamp, a cosine learning-rate schedule, and a hard squared
    operator-norm bound on the trainable A/Z coordinate.
    """

    layers: List[int]
    fact_token: Literal["subject_last"]
    v_num_grad_steps: int
    v_lr: float
    ks_bs: int
    v_loss_layer: int
    v_weight_decay: float
    kl_factor: float

    mom2_update_weight: List[float]
    rewrite_module_tmp: str
    layer_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    joint_z_micro_batch_size: int
    joint_spectral_z_op_bound: float = 25.0
    joint_rewrite_bare_loss_alpha: float = 0.5

    # Runtime state; this is not part of the method configuration.
    only_save_zs: bool = False

    def configure_rewrite_loss_mix(self) -> None:
        """Validate the hparams-defined bare/context rewrite-loss mixture."""
        alpha = float(self.joint_rewrite_bare_loss_alpha)
        if not isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError(
                "joint_rewrite_bare_loss_alpha must be finite and satisfy "
                "0 <= value <= 1"
            )
        self.joint_rewrite_bare_loss_alpha = alpha

    def configure_z_constraints(self) -> None:
        """Validate the fixed hard operator constraint."""
        bound = float(self.joint_spectral_z_op_bound)
        if not isfinite(bound) or bound <= 0.0:
            raise ValueError(
                "joint_spectral_z_op_bound must be finite and positive"
            )
        self.joint_spectral_z_op_bound = bound
