from typing import ClassVar, Literal, Protocol

import torch
from vllm.model_executor.models.interfaces import SupportsEagleBase


class EagleModelMixin:
    aux_hidden_state_layers: tuple[int, ...] = ()

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.aux_hidden_state_layers = layers

    def _maybe_add_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> list[torch.Tensor]:
        if layer_idx in self.aux_hidden_state_layers:
            value = hidden_states + residual if residual is not None else hidden_states
            aux_hidden_states.append(value)
        return aux_hidden_states


class SupportsEagle3(SupportsEagleBase, Protocol):
    """The interface required for models that support
    EAGLE-3 speculative decoding."""

    supports_eagle3: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports EAGLE-3
    speculative decoding.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """
        Set which layers should output auxiliary hidden states for EAGLE-3.

        Args:
            layers: Tuple of layer indices that should output auxiliary
                hidden states.
        """
        parent_ref = self
        if hasattr(self, "get_language_model"):
            parent_ref = self.get_language_model()
        elif hasattr(self, "language_model"):
            parent_ref = self.language_model
        assert hasattr(
            parent_ref, "model"
        ), "Model instance must have 'model' attribute to set number of layers"
        assert isinstance(
            parent_ref.model, EagleModelMixin
        ), "Model instance must inherit from EagleModelMixin to set auxiliary layers"
        parent_ref.model._set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """
        Get the default layer indices that should output auxiliary hidden states
        for EAGLE-3 for this model. Models can override this method to provide
        different default layers based on their architecture, but it is encouraged
        to instead include the layer specification in the model's config if possible.

        Returns:
            Tuple of layer indices for auxiliary hidden state outputs.
        """
        parent_ref = self
        if hasattr(self, "get_language_model"):
            parent_ref = self.get_language_model()
        elif hasattr(self, "language_model"):
            parent_ref = self.language_model
        assert hasattr(
            parent_ref, "model"
        ), "Model instance must have 'model' attribute to get number of layers"
        assert hasattr(
            parent_ref.model, "layers"
        ), "Model instance must have 'layers' attribute to get number of layers"
        num_layers = len(parent_ref.model.layers)
        return (2, num_layers // 2, num_layers - 3)
