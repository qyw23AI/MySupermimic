"""Scalar names shared by online training and existing report exports."""

UPDATE_TAGS = {
    "world_model_loss": "Loss/world_model",
    "critic_loss": "Loss/critic",
    "estimator_loss": "Loss/HIM_estimation",
    "swap_loss": "Loss/HIM_swap",
    "imagined_8_step_return": "Imagine/return_8_steps",
    "actor_objective": "Imagine/actor_objective",
    "reward_grad_norm": "Imagine/reward_grad_norm",
    "tail_grad_norm": "Imagine/tail_grad_norm",
    "tail_grad_cosine": "Imagine/tail_grad_cosine",
    "tail_grad_multiplier": "Imagine/tail_grad_multiplier",
    "actor_anchor_rmse": "Imagine/actor_anchor_rmse",
    "actor_grad_norm": "Imagine/actor_grad_norm",
    "imagined_survival_at_8": "Imagine/survival_at_8",
    "real_reward_per_step": "Train/real_reward_per_step",
    "collection_seconds": "Perf/collection_seconds",
    "update_seconds": "Perf/update_seconds",
    "elapsed_seconds": "Perf/elapsed_seconds",
    "collection_transitions_per_second": "Perf/collection_transitions_per_second",
    "torch_peak_reserved_mib": "Perf/torch_peak_reserved_mib",
    "training_transitions": "Progress/training_transitions",
    "physics_substeps": "Progress/physics_substeps",
}

EVAL_TAGS = {
    "fixed_history_action_rmse": "Eval/fixed_history_action_rmse",
    "reward_per_step": "Eval/real_reward_per_step",
    "linear_rmse_m_s": "Eval/linear_rmse_m_s",
    "yaw_rmse_rad_s": "Eval/yaw_rmse_rad_s",
    "falls": "Eval/falls",
    "falls_per_robot_minute": "Eval/falls_per_robot_minute",
}


def log_metrics(writer, metrics, tags):
    step = metrics.get("iteration", 0)
    for key, tag in tags.items():
        if key in metrics:
            writer.add_scalar(tag, metrics[key], step)
    for key, value in metrics.get("reward_terms_per_step", {}).items():
        writer.add_scalar(f"EvalReward/{key}", value, step)
