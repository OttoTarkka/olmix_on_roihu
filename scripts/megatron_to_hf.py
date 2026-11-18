#!/usr/bin/env python3
"""Convert a Megatron-LM ``torch_dist`` checkpoint of a dense GPT model to a HuggingFace Qwen3 model.

This replaces the LUMI route (Megatron-Bridge with a dummy HF model and a
templated ``run_config.yaml``). Everything about the architecture is read from
the training arguments saved in the checkpoint (``common.pt``), so the
normalisation epsilon, vocabulary padding, RoPE base and tying are exactly
those used in training. The LUMI template, for example, hard-coded
``layernorm_epsilon: 1.0e-06`` while the proxies trained with Megatron's
default of 1e-5.

Supported: the Qwen3-style models that ``scripts/train-0.05B.sh`` trains, i.e.
RMSNorm, SwiGLU, RoPE (not interleaved, full head dimension), QK-layernorm,
no linear biases, optional GQA, tied or untied embeddings, tensor and pipeline
parallel size 1. Anything else is rejected rather than converted wrongly.

Weight mapping (Megatron Core 0.16, Transformer Engine layer names; layers are
stacked on the first axis of each tensor in the checkpoint):

    embedding.word_embeddings.weight                 model.embed_tokens.weight
    decoder.layers.self_attention.linear_qkv
        .layer_norm_weight                            layers.N.input_layernorm.weight
        .weight  [ng, (np/ng + 2) * hn, h] grouped    layers.N.self_attn.{q,k,v}_proj.weight
    decoder.layers.self_attention.q_layernorm.weight layers.N.self_attn.q_norm.weight
    decoder.layers.self_attention.k_layernorm.weight layers.N.self_attn.k_norm.weight
    decoder.layers.self_attention.linear_proj.weight layers.N.self_attn.o_proj.weight
    decoder.layers.mlp.linear_fc1
        .layer_norm_weight                            layers.N.post_attention_layernorm.weight
        .weight  [gate; up] (silu(first) * second)    layers.N.mlp.{gate,up}_proj.weight
    decoder.layers.mlp.linear_fc2.weight             layers.N.mlp.down_proj.weight
    decoder.final_layernorm.weight                   model.norm.weight
    output_layer.weight (untied only)                lm_head.weight

The vocabulary keeps Megatron's padding (``padded_vocab_size``): Megatron's
softmax runs over the padded rows too, so keeping them gives the same
probabilities as in training.

Usage:
    python scripts/megatron_to_hf.py --checkpoint <run>/checkpoints --output-dir <hf_dir> [--require-final]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp

LOGGER = logging.getLogger("megatron_to_hf")
CONVERTER_VERSION = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="A Megatron checkpoints directory (uses latest_checkpointed_iteration.txt) or one iter_NNNNNNN directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write the HF model; must not exist.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer name or path to save with the model (default: the one the checkpoint was trained with).",
    )
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="Fail unless the checkpoint is from the last training iteration (train_iters).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Reading the checkpoint
# --------------------------------------------------------------------------


def resolve_iteration_dir(checkpoint: Path) -> Path:
    """Find the iteration directory to convert.

    Args:
        checkpoint: a checkpoints directory with ``latest_checkpointed_iteration.txt``,
            or an ``iter_NNNNNNN`` directory.

    Returns:
        The ``iter_NNNNNNN`` directory.

    Raises:
        FileNotFoundError: if neither layout is found.
    """
    if (checkpoint / ".metadata").is_file() and (checkpoint / "common.pt").is_file():
        return checkpoint
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise FileNotFoundError(f"{checkpoint} is neither an iter_* checkpoint nor has {tracker.name}")
    iteration = tracker.read_text().strip()
    if not iteration.isdigit():
        raise ValueError(f"{tracker} does not hold an iteration number: {iteration!r}")
    iteration_dir = checkpoint / f"iter_{int(iteration):07d}"
    if not (iteration_dir / ".metadata").is_file():
        raise FileNotFoundError(f"{tracker} points to {iteration_dir}, which is not a torch_dist checkpoint")
    return iteration_dir


class _Placeholder:
    """Stands in for Megatron classes in ``common.pt`` (enums the converter never uses)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args

    def __setstate__(self, state: Any) -> None:
        self.state = state

    def __repr__(self) -> str:
        return f"<Megatron object {self.args!r}>"


class _MegatronFreeUnpickler(pickle.Unpickler):
    """Unpickles ``common.pt`` without importing Megatron.

    Only classes from ``megatron.*`` modules are replaced; everything else
    (argparse, torch dtypes, ...) loads normally.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == "megatron" or module.startswith("megatron."):
            return _Placeholder
        return super().find_class(module, name)


class _PickleModule:
    """``pickle_module`` for torch.load that uses the Megatron-free unpickler."""

    Unpickler = _MegatronFreeUnpickler
    load = pickle.load


def load_common(iteration_dir: Path) -> dict[str, Any]:
    """Load ``common.pt``: the training arguments and the iteration.

    Args:
        iteration_dir: the ``iter_NNNNNNN`` directory.

    Returns:
        The common state dict (keys include ``args`` and ``iteration``).

    Raises:
        ValueError: if ``args`` is missing.
    """
    common = torch.load(iteration_dir / "common.pt", map_location="cpu", weights_only=False, pickle_module=_PickleModule)
    if not isinstance(common.get("args"), Namespace):
        raise ValueError(f"{iteration_dir / 'common.pt'} has no saved training arguments")
    return common


def check_supported(args: Namespace) -> None:
    """Reject training configurations this converter does not handle.

    Args:
        args: the saved Megatron training arguments.

    Raises:
        ValueError: listing every unsupported setting found.
    """
    expected = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "normalization": "RMSNorm",
        "swiglu": True,
        "position_embedding_type": "rope",
        "rotary_interleaved": False,
        "rotary_percent": 1.0,
        "qk_layernorm": True,
        "add_bias_linear": False,
        "add_qkv_bias": False,
        "use_rope_scaling": False,
        "multi_latent_attention": False,
        "num_experts": None,
        "transformer_impl": "transformer_engine",
        "qk_l2_norm": False,
        "attention_output_gate": False,
    }
    problems = [
        f"{name}={getattr(args, name, '<missing>')!r} (need {value!r})"
        for name, value in expected.items()
        if getattr(args, name, "<missing>") != value
    ]
    if getattr(args, "apply_layernorm_1p", False):
        problems.append("apply_layernorm_1p=True (zero-centred norms)")
    if getattr(args, "softmax_scale", None) is not None:
        problems.append(f"softmax_scale={args.softmax_scale!r} (need the default 1/sqrt(head_dim))")
    if problems:
        raise ValueError("unsupported checkpoint configuration: " + "; ".join(problems))


def load_model_tensors(iteration_dir: Path) -> dict[str, torch.Tensor]:
    """Read the model weights (not the optimizer state) from a torch_dist checkpoint.

    Args:
        iteration_dir: the ``iter_NNNNNNN`` directory.

    Returns:
        Megatron parameter name -> tensor, as stored (layers stacked on axis 0).

    Raises:
        ValueError: if the checkpoint holds model tensors this converter does not map.
    """
    metadata = dcp.FileSystemReader(str(iteration_dir)).read_metadata()
    state_dict: dict[str, torch.Tensor] = {}
    for name, item in metadata.state_dict_metadata.items():
        if not name.startswith(("embedding.", "decoder.", "output_layer.")) or "._extra_state" in name:
            continue
        if not hasattr(item, "size"):
            raise ValueError(f"model entry {name} is not a tensor")
        state_dict[name] = torch.empty(tuple(item.size), dtype=item.properties.dtype)
    dcp.load(state_dict, checkpoint_id=str(iteration_dir))
    return state_dict


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


def split_qkv(weight: torch.Tensor, num_heads: int, num_groups: int, head_dim: int) -> tuple[torch.Tensor, ...]:
    """Split Megatron's fused QKV weight into HF q, k and v projections.

    Megatron lays the rows out per query group: for each of ``num_groups``
    groups, the group's query heads, then its key head, then its value head
    (``attention.py`` get_query_key_value_tensors).

    Args:
        weight: ``[num_groups * (num_heads / num_groups + 2) * head_dim, hidden]``.
        num_heads: query heads.
        num_groups: key/value heads.
        head_dim: channels per head.

    Returns:
        The q (``[num_heads * head_dim, hidden]``), k and v (``[num_groups * head_dim, hidden]``) weights.
    """
    heads_per_group = num_heads // num_groups
    hidden = weight.shape[-1]
    expected_rows = num_groups * (heads_per_group + 2) * head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(f"QKV weight has {weight.shape[0]} rows, expected {expected_rows}")
    grouped = weight.view(num_groups, (heads_per_group + 2) * head_dim, hidden)
    q_rows = heads_per_group * head_dim
    query = grouped[:, :q_rows].reshape(num_heads * head_dim, hidden)
    key = grouped[:, q_rows : q_rows + head_dim].reshape(num_groups * head_dim, hidden)
    value = grouped[:, q_rows + head_dim :].reshape(num_groups * head_dim, hidden)
    return query.clone(), key.clone(), value.clone()


def convert_state_dict(megatron: dict[str, torch.Tensor], args: Namespace) -> dict[str, torch.Tensor]:
    """Map Megatron weights to HF Qwen3 parameter names.

    Args:
        megatron: the tensors from :func:`load_model_tensors`.
        args: the saved training arguments.

    Returns:
        HF state dict for ``Qwen3ForCausalLM`` (``lm_head.weight`` only if untied).

    Raises:
        ValueError: on missing or unused Megatron tensors, or wrong shapes.
    """
    num_layers = args.num_layers
    num_heads = args.num_attention_heads
    num_groups = args.num_query_groups if args.group_query_attention else num_heads
    head_dim = args.kv_channels
    ffn = args.ffn_hidden_size
    remaining = dict(megatron)

    def take(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if name not in remaining:
            raise ValueError(f"checkpoint has no {name}")
        tensor = remaining.pop(name)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
        return tensor

    hidden = args.hidden_size
    vocab = args.padded_vocab_size
    qkv_rows = num_groups * (num_heads // num_groups + 2) * head_dim
    hf: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": take("embedding.word_embeddings.weight", (vocab, hidden)),
        "model.norm.weight": take("decoder.final_layernorm.weight", (hidden,)),
    }
    layers = "decoder.layers."
    stacked = {
        "input_layernorm": take(layers + "self_attention.linear_qkv.layer_norm_weight", (num_layers, hidden)),
        "qkv": take(layers + "self_attention.linear_qkv.weight", (num_layers, qkv_rows, hidden)),
        "q_norm": take(layers + "self_attention.q_layernorm.weight", (num_layers, head_dim)),
        "k_norm": take(layers + "self_attention.k_layernorm.weight", (num_layers, head_dim)),
        "o_proj": take(layers + "self_attention.linear_proj.weight", (num_layers, hidden, num_heads * head_dim)),
        "post_attention_layernorm": take(layers + "mlp.linear_fc1.layer_norm_weight", (num_layers, hidden)),
        "fc1": take(layers + "mlp.linear_fc1.weight", (num_layers, 2 * ffn, hidden)),
        "down_proj": take(layers + "mlp.linear_fc2.weight", (num_layers, hidden, ffn)),
    }
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}."
        query, key, value = split_qkv(stacked["qkv"][layer], num_heads, num_groups, head_dim)
        gate, up = stacked["fc1"][layer].chunk(2, dim=0)  # swiglu: silu(first half) * second half
        hf.update(
            {
                prefix + "input_layernorm.weight": stacked["input_layernorm"][layer].clone(),
                prefix + "self_attn.q_proj.weight": query,
                prefix + "self_attn.k_proj.weight": key,
                prefix + "self_attn.v_proj.weight": value,
                prefix + "self_attn.q_norm.weight": stacked["q_norm"][layer].clone(),
                prefix + "self_attn.k_norm.weight": stacked["k_norm"][layer].clone(),
                prefix + "self_attn.o_proj.weight": stacked["o_proj"][layer].clone(),
                prefix + "post_attention_layernorm.weight": stacked["post_attention_layernorm"][layer].clone(),
                prefix + "mlp.gate_proj.weight": gate.clone(),
                prefix + "mlp.up_proj.weight": up.clone(),
                prefix + "mlp.down_proj.weight": stacked["down_proj"][layer].clone(),
            }
        )
    if args.untie_embeddings_and_output_weights:
        hf["lm_head.weight"] = take("output_layer.weight", (vocab, hidden))
    if remaining:
        raise ValueError(f"checkpoint tensors the converter does not map: {sorted(remaining)}")
    return hf


def build_config(args: Namespace, tokenizer: Any) -> Any:
    """The HF Qwen3 config matching the training arguments.

    Args:
        args: the saved training arguments.
        tokenizer: the HF tokenizer the model was trained with.

    Returns:
        A ``transformers.Qwen3Config``.
    """
    from transformers import Qwen3Config

    num_groups = args.num_query_groups if args.group_query_attention else args.num_attention_heads
    return Qwen3Config(
        vocab_size=args.padded_vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.ffn_hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=num_groups,
        head_dim=args.kv_channels,
        hidden_act="silu",
        max_position_embeddings=args.max_position_embeddings,
        initializer_range=args.init_method_std,
        rms_norm_eps=args.layernorm_epsilon,
        tie_word_embeddings=not args.untie_embeddings_and_output_weights,
        rope_parameters={"rope_type": "default", "rope_theta": float(args.rotary_base)},
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        dtype="bfloat16",
    )


def check_tokenizer(tokenizer: Any, args: Namespace) -> None:
    """Check the tokenizer is the one whose vocabulary Megatron padded.

    Megatron pads the vocabulary of ``len(tokenizer)`` up to a multiple of
    ``make_vocab_size_divisible_by``; a different tokenizer would not give the
    checkpoint's ``padded_vocab_size``.

    Raises:
        ValueError: if the sizes do not match.
    """
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    padded = math.ceil(len(tokenizer) / multiple) * multiple
    if padded != args.padded_vocab_size:
        raise ValueError(
            f"tokenizer has {len(tokenizer)} tokens (padded {padded}), but the checkpoint's vocabulary "
            f"is {args.padded_vocab_size}: wrong tokenizer?"
        )


def main() -> None:
    """Convert one checkpoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Hub request otherwise
    cli = parse_args()
    if cli.output_dir.exists():
        raise FileExistsError(f"{cli.output_dir} exists, not clobbering")
    # Everything is written here first and renamed at the end, so an existing
    # output directory is always a complete conversion.
    partial_dir = cli.output_dir.parent / f".{cli.output_dir.name}.partial"
    if partial_dir.exists():
        raise FileExistsError(f"{partial_dir} is left from an interrupted conversion; remove it first")

    from transformers import AutoTokenizer, Qwen3ForCausalLM

    iteration_dir = resolve_iteration_dir(cli.checkpoint)
    common = load_common(iteration_dir)
    args = common["args"]
    iteration = int(common["iteration"])
    LOGGER.info("Checkpoint %s (iteration %d of %s)", iteration_dir, iteration, args.train_iters)
    if cli.require_final and iteration != args.train_iters:
        raise ValueError(f"checkpoint is from iteration {iteration}, not the final {args.train_iters}")
    check_supported(args)

    tokenizer_name = cli.tokenizer or args.tokenizer_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    check_tokenizer(tokenizer, args)

    megatron = load_model_tensors(iteration_dir)
    hf_state = convert_state_dict(megatron, args)
    config = build_config(args, tokenizer)

    # Built on the CPU with random weights, all of which are then overwritten;
    # the checks below make sure none is left over.
    model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    result = model.load_state_dict(hf_state, strict=False)
    allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise ValueError(
            f"state dict mismatch: missing {sorted(result.missing_keys)}, unexpected {sorted(result.unexpected_keys)}"
        )
    if config.tie_word_embeddings:
        model.tie_weights()
        if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
            raise RuntimeError("lm_head is not tied to the embeddings after tie_weights()")
    model.eval()

    partial_dir.mkdir(parents=True)
    model.save_pretrained(partial_dir, safe_serialization=True)
    tokenizer.save_pretrained(partial_dir)
    info = {
        "converter": "olmix_on_roihu/scripts/megatron_to_hf.py",
        "converter_version": CONVERTER_VERSION,
        "source_checkpoint": str(iteration_dir.resolve()),
        "iteration": iteration,
        "train_iters": args.train_iters,
        "tokenizer": tokenizer_name,
        "megatron_args": {
            name: getattr(args, name)
            for name in (
                "num_layers", "hidden_size", "ffn_hidden_size", "num_attention_heads", "num_query_groups",
                "group_query_attention", "kv_channels", "seq_length", "max_position_embeddings", "rotary_base",
                "layernorm_epsilon", "padded_vocab_size", "untie_embeddings_and_output_weights", "data_path",
            )
        },
    }
    (partial_dir / "conversion_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    partial_dir.rename(cli.output_dir)
    LOGGER.info("Wrote %s", cli.output_dir)


if __name__ == "__main__":
    main()
