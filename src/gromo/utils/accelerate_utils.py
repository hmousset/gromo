from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from accelerate import Accelerator

    from gromo.containers.growing_container import GrowingContainer


def prepare(
    accelerator: Accelerator,
    model: GrowingContainer,
    /,
    *args: Any,
) -> GrowingContainer | tuple[GrowingContainer, ...]:
    """Prepare a :class:`~gromo.containers.GrowingContainer` for distributed training.

    Wraps the model with ``DistributedDataParallel`` via the given
    :class:`~accelerate.Accelerator` while keeping the original
    :class:`~gromo.containers.GrowingContainer` as the user-facing object.
    After this call, forward passes are automatically routed through the DDP
    wrapper for gradient synchronisation, and growing methods
    (``compute_optimal_updates``, ``apply_change``, …) work directly on the
    model without requiring :func:`~accelerate.Accelerator.unwrap_model`.
    The training utilities also detect the accelerator automatically, so no
    ``accelerator=`` keyword is needed in the training loop.

    Parameters
    ----------
    accelerator :
        The :class:`~accelerate.Accelerator` managing the distributed
        environment.
    model :
        The :class:`~gromo.containers.GrowingContainer` to prepare.
    *args :
        Additional objects forwarded to ``accelerator.prepare()`` — typically
        the optimizer and dataloaders.

    Returns
    -------
    GrowingContainer | tuple[GrowingContainer, ...]
        The same model object (augmented in-place).  If extra args were
        provided, returns a tuple ``(model, *prepared_args)``; otherwise
        returns the model directly.

    Examples
    --------
    >>> from accelerate import Accelerator
    >>> import gromo
    >>> accelerator = Accelerator()
    >>> model, optimizer, train_dl = gromo.prepare(accelerator, model, optimizer, train_dl)
    >>> # Training loop is identical to the single-GPU version:
    >>> model.compute_optimal_updates(maximum_added_neurons=4)
    >>> model.apply_change()
    """
    all_args = (model, *args)
    prepared = accelerator.prepare(*all_args)

    if not args:
        ddp_model = prepared
        prepared_rest: tuple[Any, ...] = ()
    else:
        ddp_model = prepared[0]
        prepared_rest = tuple(prepared[1:])

    # Store the DDP wrapper and accelerator on the model.
    # GrowingContainer.__call__ routes through _ddp when set.
    model.__dict__["_ddp"] = ddp_model
    model.__dict__["_accelerator"] = accelerator

    if prepared_rest:
        return (model, *prepared_rest)
    return model
