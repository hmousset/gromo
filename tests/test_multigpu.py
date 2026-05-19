"""Multi-process integration tests for the Accelerate-backed training utilities.

Tests run entirely on CPU using the gloo backend — no GPU required.
They exercise the actual distributed-communication paths (all-reduce,
``no_sync``, etc.) via :func:`torch.multiprocessing.spawn`.

Run with::

    uv run python -m unittest tests.test_multigpu
"""

from __future__ import annotations

import os
import random
import unittest
from pathlib import Path
from typing import Callable

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
# Must be divisible by WORLD_SIZE.
N_SAMPLES = 128
IN_FEATURES = 8
OUT_FEATURES = 4
BATCH_SIZE = 16
DAG_HIDDEN = 16
CONV_CHANNELS = 4
CONV_SPATIAL = 8
# Number of simulated ranks (CPU processes).
WORLD_SIZE = 2

# Cross-rank tolerance: after all-reduce every rank holds the same bits.
RANK_TOL = 1e-5
# Single-process vs multi-process: different accumulation order → small float32 noise.
SINGLE_TOL = 1e-3

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# Helpers: dataset / model factories
# ---------------------------------------------------------------------------


def _make_dataset() -> TensorDataset:
    torch.manual_seed(SEED)
    X = torch.randn(N_SAMPLES, IN_FEATURES)
    Y = torch.randn(N_SAMPLES, OUT_FEATURES)
    return TensorDataset(X, Y)


def _make_mlp():
    from gromo.containers.growing_mlp import GrowingMLP

    torch.manual_seed(SEED)
    return GrowingMLP(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=16,
        number_hidden_layers=2,
    )


def _make_residual_mlp():
    from gromo.containers.growing_residual_mlp import GrowingResidualMLP

    torch.manual_seed(SEED)
    model = GrowingResidualMLP(
        in_features=(IN_FEATURES,),
        out_features=OUT_FEATURES,
        num_features=16,
        hidden_features=8,
        num_blocks=2,
    )
    model.set_growing_layers()
    return model


def _make_dag():
    from gromo.containers.growing_dag import GrowingDAG

    torch.manual_seed(SEED)
    dag = GrowingDAG(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        neurons=DAG_HIDDEN,
        use_bias=True,
        use_layer_norm=False,
        default_layer_type="linear",
        name="dagtest",
    )
    dag.add_node_with_two_edges(
        dag.root,
        "hidden",
        dag.end,
        node_attributes={"type": "linear", "size": DAG_HIDDEN},
    )
    return dag


def _make_conv_dataset() -> TensorDataset:
    torch.manual_seed(SEED)
    X = torch.randn(N_SAMPLES, CONV_CHANNELS, CONV_SPATIAL, CONV_SPATIAL)
    Y = torch.randn(N_SAMPLES, CONV_CHANNELS, CONV_SPATIAL, CONV_SPATIAL)
    return TensorDataset(X, Y)


def _make_conv_dag():
    from gromo.containers.growing_dag import GrowingDAG

    torch.manual_seed(SEED)
    dag = GrowingDAG(
        in_features=CONV_CHANNELS,
        out_features=CONV_CHANNELS,
        neurons=8,
        use_bias=True,
        use_layer_norm=False,
        default_layer_type="convolution",
        kernel_size=(3, 3),
        input_shape=(CONV_SPATIAL, CONV_SPATIAL),
        name="dagconv",
    )
    dag.add_node_with_two_edges(
        dag.root,
        "hidden",
        dag.end,
        node_attributes={
            "type": "convolution",
            "size": 8,
            "kernel_size": (3, 3),
            "shape": (CONV_SPATIAL, CONV_SPATIAL),
        },
        edge_attributes={"kernel_size": (3, 3)},
    )
    return dag


def _gather_max_drift(tensor, accelerator) -> float:
    """Gather a tensor from all ranks and return the max absolute deviation from rank 0."""
    flat = tensor.detach().flatten().unsqueeze(0)
    gathered = accelerator.gather(flat)
    return (gathered - gathered[0]).abs().max().item()


# ---------------------------------------------------------------------------
# Distributed harness
# ---------------------------------------------------------------------------


def _distributed_worker(rank: int, world_size: int, fn: Callable, port: int) -> None:
    """Entry-point executed in each spawned rank.

    Sets up the gloo process group via environment variables, creates a
    CPU-based :class:`~accelerate.Accelerator`, runs ``fn(accelerator)``,
    then tears down the process group.
    """
    # Ensure the project root is importable in the spawned process (needed so
    # pickle can locate this module and any gromo imports).
    import sys

    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )

    from accelerate import Accelerator
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=True)

    try:
        accelerator = Accelerator(cpu=True)
        fn(accelerator)
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def run_distributed(fn: Callable, world_size: int = WORLD_SIZE) -> None:
    """Spawn ``world_size`` CPU processes and run ``fn(accelerator)`` in each.

    Raises on any per-rank failure — the exception from the failing rank is
    re-raised directly in the test process by :func:`torch.multiprocessing.spawn`.

    Parameters
    ----------
    fn :
        Module-level callable with signature ``fn(accelerator: Accelerator)``.
        Must be picklable (i.e. a top-level function, not a closure).
    world_size :
        Number of simulated ranks (CPU processes).
    """
    # Make this module importable in child processes regardless of how pytest
    # or unittest has set up sys.path.
    old = os.environ.get("PYTHONPATH", "")
    if _PROJECT_ROOT not in old.split(os.pathsep):
        os.environ["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + old

    try:
        port = random.randint(20000, 40000)
        mp.spawn(
            _distributed_worker,
            args=(world_size, fn, port),
            nprocs=world_size,
            join=True,
        )
    finally:
        os.environ["PYTHONPATH"] = old


# ---------------------------------------------------------------------------
# Test body functions (module-level so mp.spawn can pickle them)
# ---------------------------------------------------------------------------


def _body_evaluate_model(accelerator) -> None:
    """Multi-GPU loss must match the single-process baseline on the full dataset."""
    import gromo
    from gromo.utils.training_utils import evaluate_model

    dataset = _make_dataset()
    model = _make_mlp()
    loss_fn = nn.MSELoss(reduction="mean")

    ddp_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model, ddp_dl = gromo.prepare(accelerator, model, ddp_dl)
    multi_loss, _ = evaluate_model(model, ddp_dl, loss_fn)

    if accelerator.is_main_process:
        full_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        single_loss, _ = evaluate_model(
            model, full_dl, loss_fn, device=accelerator.device
        )
        diff = abs(multi_loss - single_loss)
        assert diff < SINGLE_TOL, (
            f"evaluate_model: multi={multi_loss:.8f} single={single_loss:.8f} diff={diff:.2e}"
        )


def _body_gradient_descent(accelerator) -> None:
    """After DDP training the model weights must be identical across all ranks."""
    import gromo
    from gromo.utils.training_utils import gradient_descent

    dataset = _make_dataset()
    model = _make_mlp()
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model, dl, optimizer = gromo.prepare(accelerator, model, dl, optimizer)

    for _ in range(2):
        gradient_descent(model, dl, optimizer, None, loss_fn)

    drift = _gather_max_drift(next(model.parameters()).data, accelerator)
    assert drift < RANK_TOL, (
        f"gradient_descent: weights diverged across ranks drift={drift:.2e}"
    )


def _body_compute_statistics(accelerator) -> None:
    """Multi-process statistics loss must match the single-process baseline."""
    import gromo
    from gromo.utils.training_utils import compute_statistics

    dataset = _make_dataset()
    model = _make_residual_mlp()
    loss_fn = nn.MSELoss(reduction="sum")

    ddp_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model, ddp_dl = gromo.prepare(accelerator, model, ddp_dl)
    multi_loss, _ = compute_statistics(model, ddp_dl, loss_fn)

    if accelerator.is_main_process:
        # Fresh model (same seed) without DDP: backward() won't trigger gradient sync.
        model_sg = _make_residual_mlp()
        full_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        single_loss, _ = compute_statistics(
            model_sg, full_dl, loss_fn, device=accelerator.device
        )
        diff = abs(multi_loss - single_loss)
        assert diff < SINGLE_TOL, (
            f"compute_statistics: multi={multi_loss:.8f} single={single_loss:.8f} diff={diff:.2e}"
        )


def _body_statistics_sync(accelerator) -> None:
    """After compute_statistics, all GrowingModule statistics must be identical
    across all ranks and must match a single-process baseline."""
    import gromo
    from gromo.utils.training_utils import compute_statistics

    dataset = _make_dataset()
    loss_fn = nn.MSELoss(reduction="sum")

    # Single-process baseline (rank 0 only).
    sg_s = sg_fisher = None
    if accelerator.is_main_process:
        model_sg = _make_residual_mlp()
        full_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        compute_statistics(model_sg, full_dl, loss_fn, device=accelerator.device)
        layer_sg = model_sg._growing_layers[0]
        sg_s = layer_sg.tensor_s().clone()
        sg_fisher = layer_sg.covariance_loss_gradient().clone()

    # Multi-process — loop code is identical to single-process.
    model = _make_residual_mlp()
    ddp_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model, ddp_dl = gromo.prepare(accelerator, model, ddp_dl)
    compute_statistics(model, ddp_dl, loss_fn)

    layer = model._growing_layers[0]

    drift_s = _gather_max_drift(layer.tensor_s(), accelerator)
    drift_fisher = _gather_max_drift(layer.covariance_loss_gradient(), accelerator)
    drift_s_growth = _gather_max_drift(layer.tensor_s_growth(), accelerator)
    assert drift_s < RANK_TOL, f"tensor_s rank drift={drift_s:.2e}"
    assert drift_fisher < RANK_TOL, f"Fisher rank drift={drift_fisher:.2e}"
    assert drift_s_growth < RANK_TOL, f"tensor_s_growth rank drift={drift_s_growth:.2e}"

    if accelerator.is_main_process:
        diff_s = (layer.tensor_s() - sg_s).abs().max().item()
        diff_fisher = (layer.covariance_loss_gradient() - sg_fisher).abs().max().item()
        assert diff_s < SINGLE_TOL, f"tensor_s vs single-process diff={diff_s:.2e}"
        assert diff_fisher < SINGLE_TOL, (
            f"Fisher vs single-process diff={diff_fisher:.2e}"
        )


def _body_optimal_delta(accelerator) -> None:
    """Fisher-conditioned delta must be identical across ranks and differ from plain delta."""
    import gromo
    from gromo.utils.training_utils import compute_statistics

    dataset = _make_dataset()
    loss_fn = nn.MSELoss(reduction="sum")

    model = _make_residual_mlp()
    ddp_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model, ddp_dl = gromo.prepare(accelerator, model, ddp_dl)
    compute_statistics(model, ddp_dl, loss_fn)

    model.compute_optimal_updates(
        maximum_added_neurons=0, compute_delta=True, use_fisher=True
    )
    layer = model._growing_layers[0]
    delta_fisher = layer.optimal_delta_layer.weight.data.clone()

    drift = _gather_max_drift(delta_fisher, accelerator)
    assert drift < RANK_TOL, f"Fisher delta rank drift={drift:.2e}"

    drift_improv = _gather_max_drift(layer.parameter_update_decrease.clone(), accelerator)
    assert drift_improv < RANK_TOL, (
        f"parameter_update_decrease rank drift={drift_improv:.2e}"
    )

    model.compute_optimal_updates(
        maximum_added_neurons=0, compute_delta=True, use_fisher=False
    )
    delta_plain = layer.optimal_delta_layer.weight.data.clone()
    fisher_effect = (delta_fisher - delta_plain).norm().item()
    assert fisher_effect > 1e-6, (
        f"Fisher conditioning had no measurable effect: ||Δ||={fisher_effect:.2e}"
    )


def _body_full_growing_cycle(accelerator) -> None:
    """Full growing lifecycle: compute_statistics → compute_optimal_updates →
    dummy_select_update → apply_change; all weights must be consistent across ranks."""
    import gromo
    from gromo.utils.training_utils import compute_statistics

    dataset = _make_dataset()
    loss_fn = nn.MSELoss(reduction="sum")
    n_new = 2

    model = _make_residual_mlp()
    ddp_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model, ddp_dl = gromo.prepare(accelerator, model, ddp_dl)

    compute_statistics(model, ddp_dl, loss_fn)
    model.compute_optimal_updates(
        maximum_added_neurons=n_new, use_fisher=True, compute_delta=True
    )

    drift_eigs = _gather_max_drift(
        model._growing_layers[0].eigenvalues_extension, accelerator
    )
    assert drift_eigs < RANK_TOL, f"eigenvalues_extension rank drift={drift_eigs:.2e}"

    drift_imp = _gather_max_drift(
        model._growing_layers[0].first_order_improvement.clone(), accelerator
    )
    assert drift_imp < RANK_TOL, f"first_order_improvement rank drift={drift_imp:.2e}"

    model.dummy_select_update()
    model.apply_change(extension_size=n_new)
    model.reset_computation()

    params_cat = torch.cat([p.data.flatten() for p in model.parameters()])
    drift_params = _gather_max_drift(params_cat, accelerator)
    assert drift_params < RANK_TOL, f"parameter drift after grow={drift_params:.2e}"


def _body_dag_statistics_sync(accelerator) -> None:
    """Linear DAG: synced statistics and optimal delta must be identical across ranks."""
    import gromo
    from gromo.utils.training_utils import compute_statistics

    dataset = _make_dataset()
    loss_fn = nn.MSELoss(reduction="sum")

    model = _make_dag()
    ddp_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model, ddp_dl = gromo.prepare(accelerator, model, ddp_dl)
    compute_statistics(model, ddp_dl, loss_fn)

    end_node = model.get_node_module(model.end)
    drift_s = _gather_max_drift(end_node.tensor_s(), accelerator)
    assert drift_s < RANK_TOL, f"DAG tensor_s rank drift={drift_s:.2e}"

    model.compute_optimal_updates()

    drift_delta = _gather_max_drift(
        model.get_edge_module(model.root, model.end).optimal_delta_layer.weight.data,
        accelerator,
    )
    assert drift_delta < RANK_TOL, f"DAG edge optimal_delta rank drift={drift_delta:.2e}"


def _body_conv_dag_statistics_sync(accelerator) -> None:
    """Conv DAG: synced statistics and optimal delta must be identical across ranks."""
    import gromo
    from gromo.utils.training_utils import compute_statistics

    dataset = _make_conv_dataset()
    loss_fn = nn.MSELoss(reduction="sum")

    model = _make_conv_dag()
    ddp_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model, ddp_dl = gromo.prepare(accelerator, model, ddp_dl)
    compute_statistics(model, ddp_dl, loss_fn)

    end_node = model.get_node_module(model.end)
    drift_s = _gather_max_drift(end_node.tensor_s(), accelerator)
    assert drift_s < RANK_TOL, f"Conv DAG tensor_s rank drift={drift_s:.2e}"

    model.compute_optimal_updates()

    drift_delta = _gather_max_drift(
        model.get_edge_module(model.root, model.end).optimal_delta_layer.weight.data,
        accelerator,
    )
    assert drift_delta < RANK_TOL, (
        f"Conv DAG edge optimal_delta rank drift={drift_delta:.2e}"
    )


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------


class TestMultiProcess(unittest.TestCase):
    """Multi-process integration tests using CPU gloo backend.

    Each test spawns :data:`WORLD_SIZE` processes that communicate via the
    gloo backend, exercising the actual all-reduce and ``no_sync`` code paths
    without requiring any GPU.
    """

    def test_evaluate_model(self):
        """Multi-process evaluate_model must match the single-process baseline."""
        run_distributed(_body_evaluate_model)

    def test_gradient_descent(self):
        """After DDP gradient descent, weights are identical across all ranks."""
        run_distributed(_body_gradient_descent)

    def test_compute_statistics(self):
        """Multi-process compute_statistics must match the single-process baseline."""
        run_distributed(_body_compute_statistics)

    def test_statistics_sync(self):
        """After compute_statistics, all module statistics are consistent across ranks."""
        run_distributed(_body_statistics_sync)

    def test_optimal_delta(self):
        """Fisher-conditioned delta is identical across ranks and differs from plain delta."""
        run_distributed(_body_optimal_delta)

    def test_full_growing_cycle(self):
        """Full grow lifecycle produces consistent weights across all ranks."""
        run_distributed(_body_full_growing_cycle)

    def test_dag_statistics_sync(self):
        """Linear DAG statistics and optimal delta are consistent across ranks."""
        run_distributed(_body_dag_statistics_sync)

    def test_conv_dag_statistics_sync(self):
        """Conv DAG statistics and optimal delta are consistent across ranks."""
        run_distributed(_body_conv_dag_statistics_sync)


if __name__ == "__main__":
    unittest.main()
