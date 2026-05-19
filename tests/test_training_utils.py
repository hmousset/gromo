from contextlib import contextmanager

import torch
import torch.utils.data
from torch import nn
from torchmetrics import Metric

import gromo
from gromo.containers.growing_container import GrowingContainer, GrowingModel
from gromo.utils.training_utils import (
    AverageMeter,
    compute_statistics,
    enumerate_dataloader,
    evaluate_model,
    gradient_descent,
)
from tests.torch_unittest import TorchTestCase


class _FakeAccelerator:
    """Single-process Accelerator stub for unit-testing the accelerator paths."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.backward_calls = 0
        self.no_sync_calls = 0
        self.reduce_calls: list[tuple[torch.Tensor, str]] = []

    def backward(self, loss: torch.Tensor) -> None:
        """Delegate to loss.backward() and record the call."""
        self.backward_calls += 1
        loss.backward()

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        """Identity reduction (single process) — records the call."""
        self.reduce_calls.append((tensor.clone(), reduction))
        return tensor

    def unwrap_model(self, model):
        """Return the model unchanged (no DDP wrapping in single-process)."""
        return model

    def prepare(self, *args):
        """Return args unchanged (no DDP wrapping in single-process)."""
        if len(args) == 1:
            return args[0]
        return args

    @contextmanager
    def no_sync(self, model):
        """No-op context manager that records usage."""
        self.no_sync_calls += 1
        yield


# ---------------------------------------------------------------------------
# Minimal test doubles for evaluate_model
# ---------------------------------------------------------------------------
class _SimpleModel(nn.Module):
    """A trivial linear model for testing evaluate_model."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x)


class _SimpleGrowingModel(GrowingModel):
    """Minimal GrowingModel whose extended_forward returns a Tensor."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features=in_features, out_features=out_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x)

    def extended_forward(self, x: torch.Tensor, mask: dict | None = None) -> torch.Tensor:
        """Extended forward returns a plain Tensor."""
        return self.forward(x)


class _SimpleGrowingContainer(GrowingContainer):
    """Minimal GrowingContainer whose extended_forward returns a tuple."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features=in_features, out_features=out_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x)

    def extended_forward(
        self, x: torch.Tensor, mask: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extended forward returns (output, None)."""
        return self.forward(x), None


class _SumMetric(Metric):
    """Accumulates the sum of first predictions — just enough to test the metrics path."""

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, _target: torch.Tensor):
        """Accumulate prediction sums."""
        self.total += preds.sum()

    def compute(self) -> torch.Tensor:
        """Return accumulated total."""
        return self.total


class TestAverageMeter(TorchTestCase):
    """Tests for AverageMeter."""

    def test_empty_meter_returns_zero(self):
        """Empty meter returns 0.0."""
        meter = AverageMeter()
        self.assertEqual(meter.compute().item(), 0.0)

    def test_float_updates(self):
        """Average of float updates is correct."""
        meter = AverageMeter()
        meter.update(torch.tensor(4.0), n=2)
        meter.update(torch.tensor(6.0), n=3)
        # sum = 4*2 + 6*3 = 26, count = 5
        self.assertAlmostEqual(meter.compute().item(), 26.0 / 5, places=6)

    def test_inf_is_skipped(self):
        """Inf values are ignored."""
        meter = AverageMeter()
        meter.update(torch.tensor(3.0))
        meter.update(torch.tensor(float("inf")))
        self.assertEqual(meter.compute().item(), 3.0)

    def test_reset(self):
        """Reset brings meter back to initial state."""
        meter = AverageMeter()
        meter.update(torch.tensor(10.0))
        meter.reset()
        self.assertEqual(meter.compute().item(), 0.0)


class TestEnumerateDataloader(TorchTestCase):
    """Tests for enumerate_dataloader."""

    @staticmethod
    def _make_dataloader(
        n_samples: int = 10,
        batch_size: int = 2,
        with_generator: bool = False,
    ) -> torch.utils.data.DataLoader:
        """Create a simple dataloader for testing."""
        x = torch.randn(n_samples, 3)
        y = torch.randint(0, 2, (n_samples,))
        dataset = torch.utils.data.TensorDataset(x, y)
        gen = torch.Generator() if with_generator else None
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, generator=gen, shuffle=True
        )

    def test_default_yields_all_batches(self):
        """Without limits, all batches are yielded."""
        dl = self._make_dataloader(n_samples=6, batch_size=2)
        batches = list(enumerate_dataloader(dl))
        self.assertEqual(len(batches), 3)

    def test_negative_bl_yields_all_batches(self):
        """Without limits, all batches are yielded."""
        dl = self._make_dataloader(n_samples=6, batch_size=2)
        batches = list(enumerate_dataloader(dl, batch_limit=-1))
        self.assertEqual(len(batches), 3)

    def test_batch_limit(self):
        """Batch limit truncates output."""
        dl = self._make_dataloader(n_samples=10, batch_size=2)
        batches = list(enumerate_dataloader(dl, batch_limit=2))
        self.assertEqual(len(batches), 2)

    def test_epochs_fraction(self):
        """Fractional epochs limits batches proportionally."""
        dl = self._make_dataloader(n_samples=10, batch_size=2)  # 5 batches
        batches = list(enumerate_dataloader(dl, epochs=0.5))
        self.assertEqual(len(batches), 2)  # int(5 * 0.5) = 2

    def test_epochs_and_batch_limit_raises(self):
        """Providing both epochs and batch_limit raises TypeError."""
        dl = self._make_dataloader()
        with self.assertRaises(TypeError):
            list(enumerate_dataloader(dl, epochs=1.0, batch_limit=5))

    def test_seed_with_generator(self):
        """Seed is applied when dataloader has a Generator."""
        dl = self._make_dataloader(with_generator=True)
        batches = list(enumerate_dataloader(dl, dataloader_seed=0))
        self.assertGreater(len(batches), 0)
        batches_again = list(enumerate_dataloader(dl, dataloader_seed=0))
        for (_, (x_1, y_1)), (_, (x_2, y_2)) in zip(batches, batches_again, strict=True):
            self.assertTrue(torch.equal(x_1, x_2))
            self.assertTrue(torch.equal(y_1, y_2))

    def test_seed_without_generator_raises(self):
        """AttributeError when seed given but no Generator."""
        dl = self._make_dataloader(with_generator=False)
        with self.assertRaises(AttributeError):
            list(enumerate_dataloader(dl, dataloader_seed=42))


class TestEvaluateModel(TorchTestCase):
    """Tests for evaluate_model."""

    @staticmethod
    def _make_dataloader(
        n_samples: int = 8,
        in_features: int = 4,
        out_features: int = 2,
        batch_size: int = 4,
    ) -> torch.utils.data.DataLoader:
        """Create a simple regression dataloader."""
        x = torch.randn(n_samples, in_features)
        y = torch.randn(n_samples, out_features)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size
        )

    def test_basic_evaluation(self):
        """Evaluate a plain nn.Module without metrics."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader()
        loss, metric_val = evaluate_model(model, dl, nn.MSELoss(reduction="mean"))
        self.assertIsInstance(loss, float)
        self.assertEqual(metric_val, 0.0)  # DummyMetric

    def test_with_metrics(self):
        """Evaluate with a custom metric (exercises the metrics branch)."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader()
        metric = _SumMetric()
        loss, metric_val = evaluate_model(
            model, dl, nn.MSELoss(reduction="mean"), metrics=metric
        )
        self.assertIsInstance(loss, float)
        self.assertIsInstance(metric_val, float)

    def test_extended_growing_model(self):
        """use_extended_model=True with a GrowingModel."""
        model = _SimpleGrowingModel(4, 2)
        dl = self._make_dataloader()
        loss, _ = evaluate_model(
            model,
            dl,
            nn.MSELoss(reduction="mean"),
            use_extended_model=True,
        )
        self.assertIsInstance(loss, float)

    def test_extended_growing_container(self):
        """use_extended_model=True with a GrowingContainer."""
        model = _SimpleGrowingContainer(4, 2)
        dl = self._make_dataloader()
        loss, _ = evaluate_model(
            model,
            dl,
            nn.MSELoss(reduction="mean"),
            use_extended_model=True,
        )
        self.assertIsInstance(loss, float)

    def test_extended_invalid_model_raises(self):
        """use_extended_model=True with a plain nn.Module raises TypeError."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader()
        with self.assertRaises(TypeError):
            evaluate_model(
                model,
                dl,
                nn.MSELoss(reduction="mean"),
                use_extended_model=True,
            )


class _FakeScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Minimal scheduler double that records step/epoch_step calls."""

    def __init__(self):
        self.step_count = 0
        self.epoch_step_count = 0

    def step(self):  # type: ignore
        """Record a step call."""
        self.step_count += 1


class TestGradientDescent(TorchTestCase):
    """Tests for gradient_descent."""

    @staticmethod
    def _make_dataloader(
        n_samples: int = 8,
        in_features: int = 4,
        out_features: int = 2,
        batch_size: int = 4,
    ) -> torch.utils.data.DataLoader:
        """Create a simple regression dataloader."""
        x = torch.randn(n_samples, in_features)
        y = torch.randn(n_samples, out_features)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size
        )

    def test_basic_training(self):
        """One round of gradient descent without scheduler or metrics."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss, metric_val = gradient_descent(
            model,
            dl,
            optimizer,
            scheduler=None,
            loss_function=nn.MSELoss(reduction="mean"),
        )
        self.assertIsInstance(loss, float)
        self.assertEqual(metric_val, 0.0)  # DummyMetric

    def test_with_metrics(self):
        """Gradient descent with a custom metric."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metric = _SumMetric()
        loss, metric_val = gradient_descent(
            model,
            dl,
            optimizer,
            scheduler=None,
            loss_function=nn.MSELoss(reduction="mean"),
            metrics=metric,
        )
        self.assertIsInstance(loss, float)
        self.assertIsInstance(metric_val, float)

    def test_with_scheduler(self):
        """Scheduler.step() called per batch, epoch_step() called once."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader(n_samples=8, batch_size=4)  # 2 batches
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with self.subTest("after_batch"):
            scheduler = _FakeScheduler()
            gradient_descent(
                model,
                dl,
                optimizer,
                scheduler=scheduler,
                loss_function=nn.MSELoss(reduction="mean"),
                scheduler_step_granularity="batch",
            )
            self.assertEqual(scheduler.step_count, 2)

        with self.subTest("after_epoch"):
            scheduler = _FakeScheduler()
            gradient_descent(
                model,
                dl,
                optimizer,
                scheduler=scheduler,
                loss_function=nn.MSELoss(reduction="mean"),
                scheduler_step_granularity="epoch",
            )
            self.assertEqual(scheduler.step_count, 1)

    def test_loss_decreases(self):
        """Loss after training is lower than before."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader(n_samples=16, batch_size=4)
        loss_fn = nn.MSELoss(reduction="mean")
        loss_before, _ = evaluate_model(model, dl, loss_fn)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(5):
            gradient_descent(
                model,
                dl,
                optimizer,
                scheduler=None,
                loss_function=loss_fn,
            )
        loss_after, _ = evaluate_model(model, dl, loss_fn)
        self.assertLess(loss_after, loss_before)


class TestComputeStatistics(TorchTestCase):
    """Tests for compute_statistics."""

    @staticmethod
    def _make_dataloader(
        n_samples: int = 8,
        in_features: int = 4,
        out_features: int = 2,
        batch_size: int = 4,
    ) -> torch.utils.data.DataLoader:
        """Create a simple regression dataloader."""
        x = torch.randn(n_samples, in_features)
        y = torch.randn(n_samples, out_features)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size
        )

    def test_basic_compute(self):
        """Compute statistics without metrics."""
        model = _SimpleGrowingContainer(4, 2)
        dl = self._make_dataloader()
        loss, metric_val = compute_statistics(
            model, dl, loss_function=nn.MSELoss(reduction="sum")
        )
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        self.assertEqual(metric_val, 0.0)  # DummyMetric

    def test_with_metrics(self):
        """Compute statistics with a custom metric."""
        model = _SimpleGrowingContainer(4, 2)
        dl = self._make_dataloader()
        metric = _SumMetric()
        loss, metric_val = compute_statistics(
            model,
            dl,
            loss_function=nn.MSELoss(reduction="sum"),
            metrics=metric,
        )
        self.assertIsInstance(loss, float)
        self.assertIsInstance(metric_val, float)


# ---------------------------------------------------------------------------
# Accelerator-path tests
# ---------------------------------------------------------------------------


class TestEvaluateModelWithAccelerator(TorchTestCase):
    """evaluate_model behaves correctly when an Accelerator stub is supplied."""

    @staticmethod
    def _make_dataloader(
        n_samples: int = 8, batch_size: int = 4
    ) -> torch.utils.data.DataLoader:
        x = torch.randn(n_samples, 4)
        y = torch.randn(n_samples, 2)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size
        )

    def test_device_is_derived_from_accelerator(self):
        """accelerator.device overrides the `device` argument."""
        model = _SimpleModel(4, 2)
        acc = _FakeAccelerator()
        acc.device = torch.device("cpu")
        loss, _ = evaluate_model(
            model,
            self._make_dataloader(),
            nn.MSELoss(reduction="mean"),
            device=torch.device("cpu"),
            accelerator=acc,
        )
        self.assertIsInstance(loss, float)

    def test_reduce_called_for_loss_meter(self):
        """accelerator.reduce is called to aggregate the loss across processes."""
        model = _SimpleModel(4, 2)
        acc = _FakeAccelerator()
        evaluate_model(
            model,
            self._make_dataloader(),
            nn.MSELoss(reduction="mean"),
            accelerator=acc,
        )
        # Two reduce calls: one for loss_meter.sum, one for loss_meter.count
        self.assertEqual(len(acc.reduce_calls), 2)
        self.assertTrue(all(r == "sum" for _, r in acc.reduce_calls))

    def test_result_matches_no_accelerator(self):
        """Single-process accelerator gives the same result as no accelerator."""
        torch.manual_seed(0)
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader()
        loss_ref, _ = evaluate_model(model, dl, nn.MSELoss(reduction="mean"))
        loss_acc, _ = evaluate_model(
            model, dl, nn.MSELoss(reduction="mean"), accelerator=_FakeAccelerator()
        )
        self.assertAlmostEqual(loss_ref, loss_acc, places=5)


class TestGradientDescentWithAccelerator(TorchTestCase):
    """gradient_descent behaves correctly when an Accelerator stub is supplied."""

    @staticmethod
    def _make_dataloader(
        n_samples: int = 8, batch_size: int = 4
    ) -> torch.utils.data.DataLoader:
        x = torch.randn(n_samples, 4)
        y = torch.randn(n_samples, 2)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size
        )

    def test_backward_delegated_to_accelerator(self):
        """accelerator.backward() is called instead of loss.backward()."""
        model = _SimpleModel(4, 2)
        acc = _FakeAccelerator()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        dl = self._make_dataloader(n_samples=8, batch_size=4)  # 2 batches

        gradient_descent(
            model,
            dl,
            optimizer,
            scheduler=None,
            loss_function=nn.MSELoss(reduction="mean"),
            accelerator=acc,
        )

        self.assertEqual(acc.backward_calls, 2)

    def test_result_matches_no_accelerator(self):
        """Single-process accelerator gives the same loss as no accelerator."""
        torch.manual_seed(0)
        model_ref = _SimpleModel(4, 2)
        torch.manual_seed(0)
        model_acc = _SimpleModel(4, 2)

        dl = self._make_dataloader()
        opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01)
        opt_acc = torch.optim.SGD(model_acc.parameters(), lr=0.01)
        loss_fn = nn.MSELoss(reduction="mean")

        loss_ref, _ = gradient_descent(model_ref, dl, opt_ref, None, loss_fn)
        loss_acc, _ = gradient_descent(
            model_acc, dl, opt_acc, None, loss_fn, accelerator=_FakeAccelerator()
        )
        self.assertAlmostEqual(loss_ref, loss_acc, places=5)


class TestComputeStatisticsWithAccelerator(TorchTestCase):
    """compute_statistics behaves correctly when an Accelerator stub is supplied."""

    @staticmethod
    def _make_dataloader(
        n_samples: int = 8, batch_size: int = 4
    ) -> torch.utils.data.DataLoader:
        x = torch.randn(n_samples, 4)
        y = torch.randn(n_samples, 2)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size
        )

    def test_no_sync_called_each_batch(self):
        """accelerator.no_sync() is entered once per batch."""
        model = _SimpleGrowingContainer(4, 2)
        acc = _FakeAccelerator()
        dl = self._make_dataloader(n_samples=8, batch_size=4)  # 2 batches

        compute_statistics(model, dl, nn.MSELoss(reduction="sum"), accelerator=acc)

        self.assertEqual(acc.no_sync_calls, 2)

    def test_backward_delegated_to_accelerator(self):
        """accelerator.backward() is called instead of loss.backward()."""
        model = _SimpleGrowingContainer(4, 2)
        acc = _FakeAccelerator()
        dl = self._make_dataloader(n_samples=8, batch_size=4)

        compute_statistics(model, dl, nn.MSELoss(reduction="sum"), accelerator=acc)

        self.assertEqual(acc.backward_calls, 2)

    def test_reduce_called_for_loss_meter(self):
        """accelerator.reduce is called to aggregate the loss meter after the loop."""
        model = _SimpleGrowingContainer(4, 2)
        acc = _FakeAccelerator()

        compute_statistics(
            model, self._make_dataloader(), nn.MSELoss(reduction="sum"), accelerator=acc
        )

        # At minimum two calls from the loss meter (sum + count)
        reduce_args = [r for _, r in acc.reduce_calls]
        self.assertGreaterEqual(reduce_args.count("sum"), 2)

    def test_result_matches_no_accelerator(self):
        """Single-process accelerator gives the same loss as no accelerator."""
        torch.manual_seed(0)
        model = _SimpleGrowingContainer(4, 2)
        dl = self._make_dataloader()
        loss_ref, _ = compute_statistics(model, dl, nn.MSELoss(reduction="sum"))

        torch.manual_seed(0)
        model2 = _SimpleGrowingContainer(4, 2)
        loss_acc, _ = compute_statistics(
            model2, dl, nn.MSELoss(reduction="sum"), accelerator=_FakeAccelerator()
        )

        self.assertAlmostEqual(loss_ref, loss_acc, places=5)


# ---------------------------------------------------------------------------
# gromo.prepare() and GrowingContainer DDP routing
# ---------------------------------------------------------------------------


class _FakeDDP:
    """Minimal DDP stand-in that records calls and forwards to the wrapped model."""

    def __init__(self, model: nn.Module):
        self.module = model
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.module(*args, **kwargs)


class TestGromoPrepareSetsAttributes(TorchTestCase):
    """gromo.prepare() stores _ddp and _accelerator on the GrowingContainer."""

    def test_model_only_returns_model(self):
        """prepare(acc, model) returns the model when no extra args are given."""
        model = _SimpleGrowingModel(4, 2)
        acc = _FakeAccelerator()
        result = gromo.prepare(acc, model)
        self.assertIs(result, model)

    def test_model_with_extra_args_returns_tuple(self):
        """prepare(acc, model, dl) returns (model, prepared_dl)."""
        model = _SimpleGrowingModel(4, 2)
        acc = _FakeAccelerator()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(4, 4), torch.randn(4, 2)),
            batch_size=2,
        )
        result = gromo.prepare(acc, model, dl)
        self.assertIsInstance(result, tuple)
        self.assertIs(result[0], model)

    def test_accelerator_stored_on_model(self):
        """After prepare, model.__dict__['_accelerator'] is the accelerator."""
        model = _SimpleGrowingModel(4, 2)
        acc = _FakeAccelerator()
        gromo.prepare(acc, model)
        self.assertIs(model.__dict__["_accelerator"], acc)

    def test_ddp_stored_on_model(self):
        """After prepare, model.__dict__['_ddp'] is the prepared (DDP-wrapped) model."""
        model = _SimpleGrowingModel(4, 2)
        acc = _FakeAccelerator()
        gromo.prepare(acc, model)
        # _FakeAccelerator.prepare returns model unchanged (single-process stub).
        self.assertIsNotNone(model.__dict__["_ddp"])

    def test_prepare_does_not_register_ddp_as_submodule(self):
        """_ddp must not appear in model.parameters() or state_dict()."""
        model = _SimpleGrowingModel(4, 2)
        acc = _FakeAccelerator()
        gromo.prepare(acc, model)
        submodule_names = {name for name, _ in model.named_modules()}
        self.assertNotIn("_ddp", submodule_names)
        self.assertNotIn("_accelerator", submodule_names)


class TestGrowingContainerDDPRouting(TorchTestCase):
    """GrowingContainer.__call__ routes through the DDP wrapper when set."""

    def test_call_routes_through_ddp(self):
        """When _ddp is set, the forward call goes through it."""
        model = _SimpleGrowingModel(4, 2)
        fake_ddp = _FakeDDP(model)
        model.__dict__["_ddp"] = fake_ddp
        x = torch.randn(2, 4)
        model(x)
        self.assertEqual(fake_ddp.calls, 1)

    def test_call_without_ddp_is_direct(self):
        """Without _ddp, the forward call goes directly to nn.Module."""
        model = _SimpleGrowingModel(4, 2)
        x = torch.randn(2, 4)
        out = model(x)
        self.assertEqual(out.shape, (2, 2))

    def test_reentrancy_flag_prevents_infinite_recursion(self):
        """DDP calling model(*args) re-enters __call__ but falls through correctly."""
        model = _SimpleGrowingModel(4, 2)

        # Simulate DDP: calling model() from inside the DDP forward triggers
        # __call__ again; the re-entrancy flag must prevent infinite recursion.
        call_count = [0]

        class _ReentrantDDP:
            def __init__(self, m):
                self.module = m

            def __call__(self, *args, **kwargs):
                call_count[0] += 1
                # Simulate DDP calling self.module(*args) — this re-enters __call__.
                return self.module(*args, **kwargs)

        model.__dict__["_ddp"] = _ReentrantDDP(model)
        x = torch.randn(2, 4)
        out = model(x)
        # The DDP wrapper was invoked once; no infinite recursion.
        self.assertEqual(call_count[0], 1)
        self.assertEqual(out.shape, (2, 2))


class TestAutoDetectAccelerator(TorchTestCase):
    """Training utilities auto-detect the accelerator from model.__dict__."""

    @staticmethod
    def _make_dataloader(
        n_samples: int = 8, batch_size: int = 4
    ) -> torch.utils.data.DataLoader:
        x = torch.randn(n_samples, 4)
        y = torch.randn(n_samples, 2)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size
        )

    def test_compute_statistics_uses_stored_accelerator(self):
        """When _accelerator is set on the model, compute_statistics uses it."""
        model = _SimpleGrowingContainer(4, 2)
        acc = _FakeAccelerator()
        model.__dict__["_accelerator"] = acc
        model.__dict__["_ddp"] = model  # simplest DDP stub: identity

        dl = self._make_dataloader()
        compute_statistics(model, dl, nn.MSELoss(reduction="sum"))

        self.assertGreater(acc.no_sync_calls, 0)
        self.assertGreater(acc.backward_calls, 0)

    def test_explicit_device_suppresses_auto_detection(self):
        """Passing device= prevents auto-detection of the stored accelerator."""
        model = _SimpleGrowingContainer(4, 2)
        acc = _FakeAccelerator()
        model.__dict__["_accelerator"] = acc

        dl = self._make_dataloader()
        compute_statistics(
            model, dl, nn.MSELoss(reduction="sum"), device=torch.device("cpu")
        )

        self.assertEqual(acc.no_sync_calls, 0)
        self.assertEqual(acc.backward_calls, 0)
