from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.utils.data
from accelerate import Accelerator
from torch import nn
from torchmetrics import Metric, classification

from gromo.containers.growing_container import GrowingContainer, GrowingModel


if TYPE_CHECKING:
    from collections.abc import Callable, Generator


# ---------------------------------------------------------------------------
# Internal helpers for transparent accelerate support
# ---------------------------------------------------------------------------


def _get_accelerator(model: Any) -> Accelerator | None:
    """Return the :class:`~accelerate.Accelerator` stored on a prepared model, if any."""
    if isinstance(model, GrowingContainer):
        return model.__dict__.get("_accelerator")
    return None


def _get_raw_model(model: Any) -> GrowingContainer:
    """Return the :class:`~gromo.containers.GrowingContainer` itself.

    With :func:`gromo.prepare` the model is always the user-facing object, so
    no unwrapping is needed.
    """
    return model  # type: ignore[return-value]


def _get_prepared_model(model: Any) -> Any:
    """Return the DDP-wrapped model stored by :func:`gromo.prepare`, if any.

    Used to pass the DDP object to ``accelerator.no_sync()``, which requires
    the wrapper rather than the inner :class:`~gromo.containers.GrowingContainer`.
    """
    if isinstance(model, GrowingContainer):
        ddp = model.__dict__.get("_ddp")
        return ddp if ddp is not None else model
    return model


class AverageMeter(object):
    """Computes and stores an average"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the meter to initial state."""
        self.sum: torch.Tensor | None = None
        self.count = 0

    def update(self, val: torch.Tensor, n: int = 1):
        """
        Updates the average with a new value.

        Parameters
        ----------
        val : torch.Tensor
            The new value to include in the average.
        n : int, optional
            The number of samples that `val` represents. Default is 1.
        """
        if torch.isfinite(val).all():
            if self.sum is None:
                self.sum = val * n
            else:
                self.sum += val * n
            self.count += n

    def compute(self) -> torch.Tensor:
        """Returns the current average.

        Returns
        -------
        torch.Tensor
            The average of the values seen so far. Returns 0.0 if no values have been
            added.
        """
        if self.count == 0:
            return torch.tensor(0.0)
            # raise ValueError("AverageMeter has no values to compute average")
        else:
            assert self.sum is not None, (
                "Sum should not be None when count is greater than 0"
            )
            return self.sum / self.count


class DummyMetric(Metric):
    """A dummy metric that always returns 0.0."""

    def __init__(self):
        super().__init__()

    def update(self, *_, **__):
        """No-op for updating the metric."""
        return

    def compute(self) -> torch.Tensor:
        """Returns the computed metric value.

        Returns
        -------
        torch.Tensor
            Always returns a tensor with value 0.0 on the device of the metric.
        """
        return torch.tensor(0.0, device=self.device)


def enumerate_dataloader(
    dataloader: torch.utils.data.DataLoader,
    dataloader_seed: int | None = None,
    batch_limit: int | None = None,
    epochs: float | None = None,
) -> Generator[tuple[int, Any]]:
    """
    A generator that yields batches from a dataloader with an optional batch limit.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader to iterate over.
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    batch_limit : int | None, optional
        Maximum number of batches to yield after `epochs` epochs.
        Use -1 for no limit. Default is None.
    epochs : float | None, optional
        Proportion of the dataloader to iterate over.
        Is incompatible with non None `batch_limit`.

    Yields
    ------
    Generator[tuple[int, Any]]
        A generator yielding tuples of (batch_index, batch_data).

    Raises
    ------
    AttributeError
        If `dataloader_seed` is provided but the dataloader does not have a random
        number generator attribute.
    TypeError
        If `epochs` and `batch_limit` are both provided.
    """
    if (epochs is not None) and (batch_limit is not None):
        msg = f"Only one  of `epochs` and `batch_limit` can be provided, but got {epochs=} and {batch_limit=}"
        raise TypeError(msg)
    assert (epochs is None) or (epochs >= 0), "Epochs must be non-negative"
    assert (batch_limit is None) or (batch_limit == -1 or batch_limit >= 0), (
        "Batch limit must be -1 or non-negative"
    )
    if dataloader_seed is not None:
        if hasattr(dataloader, "generator") and isinstance(
            dataloader.generator, torch.Generator
        ):
            dataloader.generator.manual_seed(dataloader_seed)
        else:
            raise AttributeError(
                "The dataloader does not have a 'generator' attribute of type torch.Generator, "
                "so the seed cannot be set."
            )
    if batch_limit is None:
        if epochs is None:
            batch_limit = None
        else:
            batch_limit = int(len(dataloader) * epochs)
    elif batch_limit == -1:
        batch_limit = None
    for i, batch in enumerate(dataloader):
        if batch_limit is not None and i >= batch_limit:
            break
        yield i, batch


@torch.no_grad()
def evaluate_model(
    model: nn.Module | GrowingContainer | GrowingModel,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    use_extended_model: bool = False,
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    mask: dict | None = None,
    device: torch.device | None = None,
    accelerator: Accelerator | None = None,
) -> tuple[float, float]:
    """
    Evaluate the model on a dataloader.

    Parameters
    ----------
    model : nn.Module | GrowingContainer | GrowingModel
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The dataloader for evaluation data.
    loss_function : nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        The loss function to use. Must have reduction="mean".
    use_extended_model : bool, optional
        Whether to use the extended model for evaluation. Default is False.
    metrics : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    batch_limit : int | None, optional
        Maximum number of batches to evaluate. Use -1 for no limit. Default is None.
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    mask : dict | None, optional
        The mask to use for the extended model. Only used if `use_extended_model` is True.
        Default is None.
    device : torch.device | None, optional
        Device to use.  Ignored when ``accelerator`` is provided.  When neither
        is given, an :class:`~accelerate.Accelerator` is created automatically.
        Default is None.
    accelerator : Accelerator | None, optional
        Accelerator instance for multi-GPU evaluation.  When provided, loss
        values are reduced across all processes before returning.  Detected
        automatically when the model was prepared with :func:`gromo.prepare`.
        Default is None.

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, metrics_value).

    Raises
    ------
    TypeError
        If the model is not an instance of GrowingContainer or GrowingModel when
        `use_extended_model` is True.
    """
    assert (
        not isinstance(loss_function, nn.Module) or loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    # An explicit device= is interpreted as single-process mode; skip
    # auto-detection to avoid triggering collective operations on one rank only.
    if accelerator is None and device is None:
        accelerator = _get_accelerator(model)
    if accelerator is None and device is None:
        accelerator = Accelerator()
    if accelerator is not None:
        device = accelerator.device

    # metrics meters
    loss_meter = AverageMeter()
    if metrics is None:
        metrics = DummyMetric()
    else:
        metrics.reset()
        metrics = metrics.to(device)

    # prediction function
    if use_extended_model:
        if isinstance(model, GrowingModel):
            predict_fn = lambda x: model.extended_forward(x, mask=mask)
        elif isinstance(model, GrowingContainer):
            predict_fn = lambda x: model.extended_forward(x, mask=mask)[0]
        else:
            raise TypeError(
                "Model must be an instance of GrowingModel or GrowingContainer when use_extended_model is True"
            )
    else:
        predict_fn = lambda x: model(x)

    model.eval()
    for _, (x, y) in enumerate_dataloader(
        dataloader, dataloader_seed=dataloader_seed, batch_limit=batch_limit
    ):
        x, y = x.to(device), y.to(device)
        y_pred: torch.Tensor = predict_fn(x)
        loss = loss_function(y_pred, y)
        loss_meter.update(loss, x.size(0))
        metrics.update(y_pred, y)

    if accelerator is not None:
        if loss_meter.sum is not None:
            loss_meter.sum = accelerator.reduce(loss_meter.sum, reduction="sum")
        count = torch.tensor(loss_meter.count, dtype=torch.float32, device=device)
        loss_meter.count = int(accelerator.reduce(count, reduction="sum").item())

    return loss_meter.compute().item(), metrics.compute().item()


def gradient_descent(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_function: nn.Module,
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    device: torch.device | None = None,
    scheduler_step_granularity: Literal["epoch", "batch"] = "epoch",
    accelerator: Accelerator | None = None,
) -> tuple[float, float]:
    """
    Train the model on the train_dataloader using classic gradient descent.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_dataloader : torch.utils.data.DataLoader
        The dataloader for training data.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    scheduler : torch.optim.lr_scheduler.LRScheduler | None, optional
        Learning rate scheduler. Default is None.
    loss_function : nn.Module
        The loss function to use. Must have reduction="mean".
    metrics : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    batch_limit : int | None, optional
        Maximum number of batches to train. Use -1 for no limit. Default is None.
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    device : torch.device | None, optional
        Device to use.  Ignored when ``accelerator`` is provided.  When neither
        is given, an :class:`~accelerate.Accelerator` is created automatically.
        Default is None.
    scheduler_step_granularity : Literal["epoch", "batch"], optional
        Whether to step the scheduler after each epoch (``"epoch"``, default) or each mini-batch (``"batch"``).
    accelerator : Accelerator | None, optional
        Accelerator instance for multi-GPU training.  When provided,
        ``accelerator.backward(loss)`` is used in place of ``loss.backward()``.
        Detected automatically when the model was prepared with
        :func:`gromo.prepare`. Default is None.

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, aux_loss_function_value).
    """
    assert (
        not isinstance(loss_function, nn.Module) or loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    # An explicit device= is interpreted as single-process mode; skip
    # auto-detection to avoid triggering collective operations on one rank only.
    if accelerator is None and device is None:
        accelerator = _get_accelerator(model)
    if accelerator is None and device is None:
        accelerator = Accelerator()
    if accelerator is not None:
        device = accelerator.device

    # metrics meters
    loss_meter = AverageMeter()
    if metrics is None:
        metrics = DummyMetric()
    else:
        metrics.reset()
        metrics = metrics.to(device)

    model.train()
    for i, (x, y) in enumerate_dataloader(
        train_dataloader, dataloader_seed=dataloader_seed, batch_limit=batch_limit
    ):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_function(y_pred, y)
        assert loss.isnan().sum() == 0, (
            f"During training of {model}, loss is NaN: {loss}, sample index: {i / len(train_dataloader)}"
        )

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()  # accelerator=None only when device= was passed explicitly
        optimizer.step()

        # update metrics
        loss_meter.update(loss.detach(), x.size(0))
        metrics.update(y_pred.detach(), y)

        if scheduler is not None and scheduler_step_granularity == "batch":
            scheduler.step()

    if scheduler is not None and scheduler_step_granularity == "epoch":
        scheduler.step()

    return loss_meter.compute().item(), metrics.compute().item()


def compute_statistics(
    model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = nn.MSELoss(reduction="sum"),
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    device: torch.device | None = None,
    accelerator: Accelerator | None = None,
) -> tuple[float, float]:
    """
    Compute the tensor of statistics of the model on the dataloader
    with a limit of batch_limit batches.

    Parameters
    ----------
    model : GrowingContainer
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use.
    loss_function : nn.Module
        The loss function to use. Must have reduction="sum".
    metrics : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    batch_limit : int | None, optional
        The maximum number of batches to use. Default is None (no limit).
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    device : torch.device | None, optional
        Device to use.  Ignored when ``accelerator`` is provided.  When neither
        is given, an :class:`~accelerate.Accelerator` is created automatically.
        Default is None.
    accelerator : Accelerator | None, optional
        Accelerator instance for multi-GPU statistics computation.  When
        provided, each backward pass is wrapped with ``no_sync()`` so that DDP
        does not all-reduce gradients between batches — each rank accumulates
        local statistics independently — and the tensors are all-reduced once
        after the loop.  Detected automatically when the model was prepared with
        :func:`gromo.prepare`. Default is None.

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, metrics_value).
    """
    assert not isinstance(loss_function, nn.Module) or loss_function.reduction == "sum", (
        "The loss function should not be averaged over the batch"
    )

    # An explicit device= is interpreted as single-process mode; skip
    # auto-detection to avoid triggering collective operations on one rank only.
    if accelerator is None and device is None:
        accelerator = _get_accelerator(model)
    if accelerator is None and device is None:
        accelerator = Accelerator()
    if accelerator is not None:
        device = accelerator.device

    loss_meter = AverageMeter()
    if metrics is None:
        metrics = DummyMetric()
    else:
        metrics.reset()
        metrics = metrics.to(device)

    raw_model: GrowingContainer = (
        _get_raw_model(model) if accelerator is not None else model
    )  # type: ignore[assignment]
    prepared_model = _get_prepared_model(model)  # DDP model for no_sync()

    raw_model.init_computation()
    model.eval()
    for _, (x, y) in enumerate_dataloader(
        dataloader, dataloader_seed=dataloader_seed, batch_limit=batch_limit
    ):
        model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        if accelerator is not None:
            with accelerator.no_sync(prepared_model):
                accelerator.backward(loss)
        else:
            loss.backward()
        raw_model.update_computation()
        loss_meter.update(loss.detach() / x.size(0), x.size(0))
        metrics.update(y_pred.detach(), y)

    if accelerator is not None:
        raw_model.sync_computation(accelerator)
        if loss_meter.sum is not None:
            loss_meter.sum = accelerator.reduce(loss_meter.sum, reduction="sum")
        count = torch.tensor(loss_meter.count, dtype=torch.float32, device=device)
        loss_meter.count = int(accelerator.reduce(count, reduction="sum").item())

    return loss_meter.compute().item(), metrics.compute().item()


def evaluate_extended_dataset(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    mask: dict | None = None,
    accelerator: Accelerator | None = None,
) -> tuple[float, float]:
    """Evaluate extended network on dataset

    Parameters
    ----------
    model : nn.Module
        network to evaluate
    dataloader : torch.utils.data.DataLoader
        dataloader containing the data
    loss_fn : Callable
        loss function for bottleneck calculation
    mask : dict | None, optional
        extension mask for specific nodes and edges, by default None
        example: mask["edges"] for edges and mask["nodes"] for nodes
    accelerator : Accelerator | None, optional
        Accelerator instance for distributed evaluation. When None an
        :class:`~accelerate.Accelerator` is created automatically.

    Returns
    -------
    tuple[float, float]
        accuracy and loss
    """
    _, y = next(iter(dataloader))
    nb_classes = model.out_features if (y.dim() == 1 and model.out_features > 1) else None
    metric = (
        classification.MulticlassAccuracy(model.out_features, average="micro")
        if nb_classes is not None
        else None
    )
    loss, accuracy = evaluate_model(
        model,
        dataloader,
        loss_fn,
        metrics=metric,
        use_extended_model=True,
        mask=mask,
        accelerator=accelerator,
    )
    if metric is None:
        accuracy = -1
    return accuracy, loss


def evaluate_dataset(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    accelerator: Accelerator | None = None,
) -> tuple[float, float]:
    """Evaluate network on dataset

    Parameters
    ----------
    model : nn.Module
        network to evaluate
    dataloader : torch.utils.data.DataLoader
        dataloader containing the data
    loss_fn : Callable
        loss function for bottleneck calculation
    accelerator : Accelerator | None, optional
        Accelerator instance for distributed evaluation. When None an
        :class:`~accelerate.Accelerator` is created automatically.

    Returns
    -------
    tuple[float, float]
        accuracy and loss
    """
    _, y = next(iter(dataloader))
    nb_classes = model.out_features if (y.dim() == 1 and model.out_features > 1) else None
    metric = (
        classification.MulticlassAccuracy(model.out_features, average="micro")
        if nb_classes is not None
        else None
    )
    loss, accuracy = evaluate_model(
        model, dataloader, loss_fn, metrics=metric, accelerator=accelerator
    )
    if metric is None:
        accuracy = -1
    return accuracy, loss
