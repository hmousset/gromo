from unittest import TestCase, main

import torch
from torch.utils.data import DataLoader, TensorDataset

from gromo.utils.tensor_statistic import (
    TensorStatistic,
    TensorStatiticWithEstimationError,
)
from gromo.utils.utils import reset_device, set_device


class _FakeAccelerator:
    """Single-process stub that records calls to accelerator.reduce."""

    def __init__(self, scale: float = 1.0):
        self.device = torch.device("cpu")
        self.reduce_calls: list[tuple[torch.Tensor, str]] = []
        self._scale = scale

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        self.reduce_calls.append((tensor.clone(), reduction))
        return tensor * self._scale


class TestTensorStatistic(TestCase):
    _tested_class = TensorStatistic

    def test_mean(self):
        set_device("cpu")
        x = None
        n_samples = 0
        f = lambda: (x.sum(dim=0), x.size(0))  # type: ignore
        tensor_statistic = self._tested_class(
            shape=(2, 3), update_function=f, name="Average"
        )
        tensor_statistic_unshaped = self._tested_class(
            shape=None, update_function=f, name="Average-unshaped"
        )

        for t in [tensor_statistic, tensor_statistic_unshaped]:
            self.assertRaises(ValueError, t)

        tensor_statistic.init()
        tensor_statistic_unshaped.init()
        mean_x = torch.zeros((2, 3))
        for n in [1, 5, 8, 15]:
            x = torch.randn(n, 2, 3)
            n_samples += x.size(0)
            mean_x += x.sum(dim=0)
            for t in [tensor_statistic, tensor_statistic_unshaped]:
                t.updated = False
                t.update()
                self.assertTrue(torch.allclose(t(), mean_x / n_samples))
                self.assertEqual(t.samples, n_samples)

                t.update()
                self.assertTrue(torch.allclose(t(), mean_x / n_samples))
                self.assertEqual(t.samples, n_samples)

        x = torch.zeros(1, 3, 4)
        for t in [tensor_statistic, tensor_statistic_unshaped]:
            t.updated = False
            self.assertRaises(AssertionError, t.update)

            t.reset()
            self.assertIsNone(t._tensor)
            self.assertEqual(t.samples, 0)

    def tearDown(self) -> None:
        reset_device()


class TestTensorStatiticWithEstimationError(TestTensorStatistic):
    _tested_class = TensorStatiticWithEstimationError

    def setUp(self) -> None:
        set_device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)

    def test_error(self):
        num_batches = 10
        batch_size = 10
        total_samples = torch.Size((num_batches * batch_size,))
        mean = torch.tensor([3.0, 4.0])
        cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])

        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        samples = dist.sample(total_samples)
        dataloader = DataLoader(
            TensorDataset(samples), batch_size=batch_size, shuffle=False
        )

        mean_statistic = TensorStatiticWithEstimationError(
            shape=None,
            update_function=lambda x: (x.sum(dim=0), x.size(0)),
            name="Mean with Error",
        )

        self.assertRaises(AssertionError, mean_statistic.error)
        for i, (batch,) in enumerate(dataloader):
            mean_statistic.updated = False
            mean_statistic.update(x=batch)
            if i == 0:
                self.assertEqual(mean_statistic.error(), float("inf"))
        self.assertEqual(mean_statistic.samples, num_batches * batch_size)
        true_error = torch.norm(mean_statistic() - mean).item() ** 2
        self.assertLessEqual(
            true_error, mean_statistic.error() * 3
        )  # this test pass most of the time, but can fail due to randomness
        # (if no seed is set)

        cov_statistic = TensorStatiticWithEstimationError(
            shape=None,
            update_function=lambda x: (
                (x - mean_statistic()).T @ (x - mean_statistic()),
                x.size(0),
            ),
            name="Covariance with Error",
        )

        for (batch,) in dataloader:
            cov_statistic.updated = False
            cov_statistic.update(x=batch)

        true_error = torch.norm(cov_statistic() - cov).item() ** 2
        self.assertLessEqual(
            true_error, cov_statistic.error() * 2
        )  # this test pass most of the time, but can fail due to randomness
        # (if no seed is set)

    def test_stop_trace_computation(self):
        num_batches = 3
        batch_size = 10
        total_samples = torch.Size((num_batches * batch_size,))
        mean = torch.tensor([3.0, 4.0])
        cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])

        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        samples = dist.sample(total_samples)
        dataloader = DataLoader(
            TensorDataset(samples), batch_size=batch_size, shuffle=False
        )

        mean_statistic = TensorStatiticWithEstimationError(
            shape=None,
            update_function=lambda x: (x.sum(dim=0), x.size(0)),
            name="Mean with Error",
            trace_precision=5.0,
        )

        self.assertRaises(AssertionError, mean_statistic.error)
        for i, (batch,) in enumerate(dataloader):
            mean_statistic.updated = False
            mean_statistic.update(x=batch)

        self.assertFalse(mean_statistic._compute_trace)


class TestTensorStatisticSync(TestCase):
    """Tests for TensorStatistic.sync()."""

    def _make_statistic(self, values: list[torch.Tensor]) -> TensorStatistic:
        """Build a TensorStatistic pre-loaded with a sequence of tensors.

        Each tensor is cloned inside the update function so that TensorStatistic's
        in-place ``+=`` accumulation does not mutate the original tensors in ``values``.
        """
        data = iter(values)
        stat = TensorStatistic(
            shape=None,
            update_function=lambda: (next(data).clone(), 1),
            name="test",
        )
        for _ in values:
            stat.updated = False
            stat.update()
        return stat

    def test_sync_reduces_tensor_and_samples(self):
        """sync() calls reduce on _tensor and updates samples."""
        stat = self._make_statistic([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])])
        acc = _FakeAccelerator(scale=1.0)

        stat.sync(acc)

        # reduce called for _tensor and for the samples count
        self.assertEqual(len(acc.reduce_calls), 2)
        self.assertEqual(acc.reduce_calls[0][1], "sum")
        self.assertEqual(acc.reduce_calls[1][1], "sum")
        self.assertEqual(stat.samples, 2)

    def test_sync_scales_tensor_by_accelerator(self):
        """sync() honours the reduction returned by the accelerator (scale=2 simulates 2 GPUs)."""
        stat = self._make_statistic([torch.tensor([1.0, 2.0])])
        acc = _FakeAccelerator(scale=2.0)

        stat.sync(acc)

        # _tensor was multiplied by 2 (as if a second GPU had the same local sum)
        self.assertTrue(torch.allclose(stat._tensor, torch.tensor([2.0, 4.0])))
        # samples count was also doubled
        self.assertEqual(stat.samples, 2)

    def test_sync_on_empty_statistic(self):
        """sync() on an uninitialised statistic only reduces the zero count."""
        stat = TensorStatistic(shape=None, update_function=lambda: (torch.zeros(2), 0))
        acc = _FakeAccelerator(scale=1.0)

        stat.sync(acc)

        # _tensor is None → only one reduce call (for samples)
        self.assertEqual(len(acc.reduce_calls), 1)
        self.assertIsNone(stat._tensor)

    def test_sync_is_consistent_with_manual_split(self):
        """Splitting data across two fake GPUs and syncing gives the same final average."""
        # Use exact integer values so float32 addition is associative
        t0 = torch.tensor([1.0, 2.0, 3.0])
        t1 = torch.tensor([4.0, 5.0, 6.0])

        # Single-GPU: accumulates both tensors
        single = self._make_statistic([t0, t1])
        single_sum = single._tensor.clone()
        single_n = single.samples

        # Two-GPU split: each sees one tensor
        gpu0 = self._make_statistic([t0])
        gpu1 = self._make_statistic([t1])

        # Merge via all-reduce(sum)
        merged_sum = gpu0._tensor + gpu1._tensor
        merged_n = gpu0.samples + gpu1.samples

        self.assertTrue(torch.allclose(merged_sum, single_sum))
        self.assertEqual(merged_n, single_n)


if __name__ == "__main__":
    main()
