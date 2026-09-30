from __future__ import annotations

import torch
from torch import nn

from t5gemma2_vllm_plugin.vllm_dflare import DFlareDraftModel


class _FakeDFlareInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 3)
        self.loaded_names: list[str] = []

    def load_weights(self, weights) -> set[str]:
        items = list(weights)
        self.loaded_names.extend(name for name, _ in items)
        return set(self.loaded_names)


def test_dflare_loader_marks_own_embedding_and_lm_head() -> None:
    # Construct only the small part of the object needed by load_weights. A
    # full DFlare model requires a distributed vLLM device configuration.
    model = DFlareDraftModel.__new__(DFlareDraftModel)
    nn.Module.__init__(model)
    model.model = _FakeDFlareInnerModel()
    model.lm_head = nn.Linear(3, 2, bias=False)
    model.has_own_embed_tokens = False
    model.has_own_lm_head = False

    model.load_weights(
        [
            ("embed_tokens.weight", torch.ones(4, 3)),
            ("lm_head.weight", torch.ones(2, 3)),
        ]
    )

    assert model.has_own_embed_tokens
    assert model.has_own_lm_head
    assert model.model.loaded_names == ["embed_tokens.weight"]
    torch.testing.assert_close(model.lm_head.weight, torch.ones(2, 3))


def _encoder_without_vision_loader(text_only_mode: bool):
    from t5gemma2_vllm_plugin.t5gemma2_encoder import T5Gemma2Encoder

    encoder = T5Gemma2Encoder.__new__(T5Gemma2Encoder)
    nn.Module.__init__(encoder)
    encoder.vision_tower = nn.Identity()
    encoder.text_only_mode = text_only_mode
    return encoder


def test_encoder_requires_opt_in_when_vision_weights_cannot_be_loaded() -> None:
    import pytest

    encoder = _encoder_without_vision_loader(text_only_mode=False)

    with pytest.raises(RuntimeError, match="T5GEMMA2_TEXT_ONLY=1"):
        encoder.load_weights(
            [("vision_tower.proj.weight", torch.ones(2, 2))]
        )


def test_encoder_skips_vision_weights_with_explicit_text_only_opt_in() -> None:
    encoder = _encoder_without_vision_loader(text_only_mode=True)

    loaded = encoder.load_weights(
        [("vision_tower.proj.weight", torch.ones(2, 2))]
    )

    assert loaded == set()


def test_encoder_uses_existing_vision_loader() -> None:
    class VisionLoader(nn.Module):
        def load_weights(self, weights) -> set[str]:
            assert list(weights)[0][0] == "proj.weight"
            return {"proj.weight"}

    encoder = _encoder_without_vision_loader(text_only_mode=False)
    encoder.vision_tower = VisionLoader()

    loaded = encoder.load_weights(
        [("vision_tower.proj.weight", torch.ones(2, 2))]
    )

    assert loaded == {"vision_tower.proj.weight"}


def test_encoder_rejects_images_in_text_only_mode() -> None:
    import pytest

    encoder = _encoder_without_vision_loader(text_only_mode=True)

    with pytest.raises(ValueError, match="Image inputs are disabled"):
        encoder.get_image_features(torch.ones(1, 3, 4, 4))
