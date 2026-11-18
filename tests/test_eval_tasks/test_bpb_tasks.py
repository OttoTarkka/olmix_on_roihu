"""Checks on the BPB task definitions in eval_tasks/bpb that run without lm-eval.

They cover two things lm-eval would only reveal at evaluation time on a GPU:

* the scoring functions added for olmix_on_roihu (Okapi HellaSwag, ARC, MMLU and
  Global PIQA) pick the right answer and emit the (loglikelihood, bytes) pair
  that lm-eval's ``bits_per_byte`` aggregation expects, and fail on malformed
  documents instead of skipping them;
* every task YAML resolves: its ``include`` exists, its ``!function`` names exist
  in utils.py, and every task in the olmix_bpb_<lang> groups is defined once.

The document fixtures copy the field layout of the Hugging Face datasets
(checked against the dataset viewer on 2026-11-18).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

TASK_DIR = Path(__file__).resolve().parents[2] / "eval_tasks" / "bpb"
LANGUAGE_GROUPS = ("olmix_bpb_deu", "olmix_bpb_spa", "olmix_bpb_hun")


def _load_utils() -> ModuleType:
    """Import eval_tasks/bpb/utils.py the way lm-eval does, by file path."""
    spec = importlib.util.spec_from_file_location("bpb_utils", TASK_DIR / "utils.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = _load_utils()


class _ListDataset:
    """Minimal stand-in for datasets.Dataset: just the ``map`` that process_docs uses."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def map(self, function: Any) -> list[dict[str, Any]]:
        """Merge each function result into its document, like datasets.Dataset.map."""
        return [{**doc, **function(doc)} for doc in self.docs]


def _results(*loglikelihoods: float) -> list[tuple[float, bool]]:
    """lm-eval's multiple-choice payload: one (loglikelihood, is_greedy) per choice."""
    return [(ll, False) for ll in loglikelihoods]


def _bits_per_byte(pairs: list[tuple[float, int]]) -> float:
    """lm-eval's bits_per_byte aggregation (lm_eval/api/metrics.py)."""
    return -sum(ll for ll, _ in pairs) / sum(n for _, n in pairs) / math.log(2)


# --------------------------------------------------------------------------
# Scoring functions
# --------------------------------------------------------------------------


def _okapi_arc_doc(**overrides: Any) -> dict[str, Any]:
    doc = {
        "id": "ARC-Challenge/test/Mercury_SC_415719",
        "answer": "C",
        "instruction": "Miért olvad el a jégkocka?",
        "option_a": "A hő a kezéből áramlik.",
        "option_b": "A hideg a kezéből áramlik.",
        "option_c": "A hő a jégkockából áramlik.",
        "option_d": "A hideg a jégkockából áramlik.",
        "option_e": None,
    }
    doc.update(overrides)
    return doc


def test_okapi_arc_drops_trailing_empty_option_and_scores_gold() -> None:
    (doc,) = utils.process_docs_okapi_arc(_ListDataset([_okapi_arc_doc()]))
    assert doc["choices"] == [
        "A hő a kezéből áramlik.",
        "A hideg a kezéből áramlik.",
        "A hő a jégkockából áramlik.",
        "A hideg a jégkockából áramlik.",
    ]
    assert doc["gold"] == 2
    assert doc["query"] == "Question: Miért olvad el a jégkocka?\nAnswer:"

    scores = utils.process_results_okapi_arc(doc, _results(-5.0, -4.0, -3.0, -6.0))
    gold_bytes = len("A hő a jégkockából áramlik.".encode("utf-8"))
    assert scores["bits_per_byte"] == (-3.0, gold_bytes)
    assert scores["acc"] == 1.0


def test_okapi_arc_keeps_fifth_option() -> None:
    (doc,) = utils.process_docs_okapi_arc(_ListDataset([_okapi_arc_doc(option_e="Egyik sem.", answer="E")]))
    assert len(doc["choices"]) == 5
    assert doc["gold"] == 4


def test_okapi_arc_rejects_gap_in_options() -> None:
    with pytest.raises(ValueError, match="gap"):
        utils.process_docs_okapi_arc(_ListDataset([_okapi_arc_doc(option_c="", option_e="Egyik sem.")]))


def test_okapi_arc_rejects_answer_beyond_options() -> None:
    with pytest.raises(ValueError, match="not one of"):
        utils.process_docs_okapi_arc(_ListDataset([_okapi_arc_doc(answer="E")]))


def test_okapi_mmlu_is_cloze_and_scores_gold_text() -> None:
    doc = {
        "instruction": "Melyik két forgatókönyv esetében?",
        "option_a": "Helytelen, helytelen",
        "option_b": "Helytelen, nem helytelen",
        "option_c": "Nem helytelen, helytelen",
        "option_d": "Nem helytelen, nem helytelen",
        "answer": "B",
        "id": "moral_scenarios/test/106",
    }
    assert utils.doc_to_choice_okapi_mmlu(doc)[1] == "Helytelen, nem helytelen"
    assert utils.doc_to_target_okapi_mmlu(doc) == 1
    scores = utils.process_results_okapi_mmlu(doc, _results(-9.0, -7.5, -8.0, -8.5))
    assert scores["bits_per_byte"] == (-7.5, len("Helytelen, nem helytelen".encode("utf-8")))
    assert scores["acc"] == 1.0


def test_okapi_hellaswag_uses_hellaswag_functions() -> None:
    raw = {
        "ctx_a": "[header] Hogyan készítsünk frizurát [title] Mosd meg a hajad.",
        "ctx_b": "",
        "activity_label": "Personal Care and Style",
        "endings": ["Első vég.", "Második vég.", "Harmadik vég.", "Negyedik vég."],
        "label": "2",
    }
    (doc,) = utils.process_docs_hellaswag(_ListDataset([raw]))
    assert doc["gold"] == 2
    assert "[" not in doc["query"]
    scores = utils.process_results_hellaswag(doc, _results(-4.0, -4.0, -2.0, -4.0))
    assert scores["bits_per_byte"] == (-2.0, len("Harmadik vég.".encode("utf-8")))


def test_global_piqa_scores_gold_solution() -> None:
    doc = {
        "prompt": "Egy műanyag zacskót megtöltenek levegővel.",
        "solution0": "Ugyanannyi marad",
        "solution1": "Csökken",
        "solution2": "Nő",
        "solution3": "Megváltozik a színe",
        "label": 3,
    }
    scores = utils.process_results_global_piqa(doc, _results(-3.0, -3.0, -3.0, -1.5))
    assert scores["bits_per_byte"] == (-1.5, len("Megváltozik a színe".encode("utf-8")))
    assert scores["acc"] == 1.0


def test_score_rejects_unresolvable_gold() -> None:
    with pytest.raises(ValueError, match="Malformed document"):
        utils._score(_results(-1.0, -2.0), ["a", "b"], 5)


def test_score_rejects_wrong_number_of_loglikelihoods() -> None:
    with pytest.raises(ValueError, match="Malformed document"):
        utils._score(_results(-1.0), ["a", "b"], 0)


def test_pairs_aggregate_to_byte_weighted_bpb() -> None:
    first = utils._score(_results(-2.0, -8.0), ["ab", "x"], 0)["bits_per_byte"]
    second = utils._score(_results(-1.0, -12.0), ["y", "abcdef"], 1)["bits_per_byte"]
    assert first == (-2.0, 2) and second == (-12.0, 6)
    # Byte-weighted: (2 + 12) nats over (2 + 6) bytes = 1.75 nats per byte.
    # A mean of per-document ratios would give (1 + 2) / 2 = 1.5 instead.
    assert _bits_per_byte([first, second]) == pytest.approx(1.75 / math.log(2))


# --------------------------------------------------------------------------
# YAML structure
# --------------------------------------------------------------------------


class _TaskLoader(yaml.SafeLoader):
    """YAML loader that keeps lm-eval's ``!function`` tags as plain strings."""


_TaskLoader.add_constructor("!function", lambda loader, node: ("!function", loader.construct_scalar(node)))


def _task_files() -> list[Path]:
    """Task and template YAMLs: ``*.yaml`` plus the ``_*_yaml`` templates lm-eval includes."""
    return sorted(
        path
        for path in TASK_DIR.rglob("*")
        if path.is_file() and (path.suffix == ".yaml" or path.name.endswith("_yaml"))
    )


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=_TaskLoader)
    assert isinstance(data, dict), f"{path} is not a YAML mapping"
    return data


def _function_names(value: Any) -> list[str]:
    if isinstance(value, tuple) and value[:1] == ("!function",):
        return [value[1]]
    if isinstance(value, dict):
        return [name for item in value.values() for name in _function_names(item)]
    if isinstance(value, list):
        return [name for item in value for name in _function_names(item)]
    return []


def test_includes_and_functions_resolve() -> None:
    files = _task_files()
    assert files, f"no task files under {TASK_DIR}"
    for path in files:
        data = _load(path)
        if "include" in data:
            assert (path.parent / data["include"]).is_file(), f"{path}: missing include {data['include']}"
        for name in _function_names(data):
            module_name, _, function_name = name.rpartition(".")
            assert module_name == "utils", f"{path}: unexpected module in !function {name}"
            assert callable(getattr(utils, function_name, None)), f"{path}: utils has no {function_name}"


def _resolved_config(path: Path) -> dict[str, Any]:
    """A task's config with its ``include`` chain merged in (the including file wins)."""
    data = _load(path)
    if "include" not in data:
        return data
    parent = _resolved_config(path.parent / data["include"])
    return {**parent, **{key: value for key, value in data.items() if key != "include"}}


def test_group_tasks_name_their_fewshot_split() -> None:
    """The swarms evaluate 5-shot, so every fit task must say where examples come from.

    Without fewshot_split, lm-eval falls back to train, validation or test in
    that order, and only removes the scored document from its own examples
    when the few-shot split equals the test split.
    """
    task_files: dict[str, Path] = {}
    for path in _task_files():
        task = _load(path).get("task")
        if isinstance(task, str):  # group files hold a list here
            task_files[task] = path

    for group in LANGUAGE_GROUPS:
        for task in _load(TASK_DIR / f"{group}.yaml")["task"]:
            config = _resolved_config(task_files[task])
            fewshot_split = config.get("fewshot_split")
            assert fewshot_split, f"{task} has no fewshot_split"
            # lm-eval evaluates on test_split, or on validation_split if there is none.
            eval_split = config.get("test_split") or config.get("validation_split")
            assert eval_split, f"{task} has no evaluation split"
            if fewshot_split == eval_split and fewshot_split != config.get("test_split"):
                pytest.fail(f"{task} takes examples from its evaluation split without declaring it test_split")


def test_language_groups_reference_defined_tasks() -> None:
    defined: dict[str, Path] = {}
    for path in _task_files():
        task = _load(path).get("task")
        if isinstance(task, str):
            assert task not in defined, f"task {task} defined in both {defined[task]} and {path}"
            defined[task] = path

    for group in LANGUAGE_GROUPS:
        data = _load(TASK_DIR / f"{group}.yaml")
        assert data["group"] == group
        assert len(data["task"]) == 6, f"{group} should list the same six benchmarks as the others"
        missing = [task for task in data["task"] if task not in defined]
        assert not missing, f"{group} lists undefined tasks: {missing}"
