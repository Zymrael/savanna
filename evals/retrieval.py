"""
Usage: python -m eval.scripts.needle_in_a_haystack \
           --haystack-max-length 9000 \
           --model-name evo \
           --device cuda:0

Tests for the ability for a DNA LM to retrieve a randomly inserted sequence (the needle)
in its context (the haystack).

Leverages the finding that, if the LM sees a repeated sequence in its context (even if
completely random), it will use the information from its context to predict the repeated
element, as quantified by the ``categorical Jacobian'' style analysis.
"""

import math
import os
import random
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import List, Optional

import numpy as np
import pandas as pd

from savanna import print_rank_0
from savanna.arguments import GlobalConfig


@contextmanager
def temporary_seed(seed: int):
    """
    Context manager for temporarily setting a random seed.

    Args:
        seed (int): The random seed to use temporarily

    Example:
        with temporary_seed(42):
            # This code will use seed 42
            print(random.random())
        # Outside the context, the original random state is restored
    """
    state = random.getstate()  # Save current state.
    random.seed(seed)  # Set temporary seed.
    try:
        yield
    finally:
        random.setstate(state)  # Restore original state.


@dataclass
class NeedleInAHaystackArgs:
    needle_length: int = 100
    haystack_min_length: int = 512
    haystack_max_length: int = 8192
    n_divisions: int = 10
    perturb_flank: int = 10
    haystack_batch_size: Optional[int] = None
    background_sequence_path: str = "eval/data/random_megabase.fasta"
    model_name: str = "evo2"
    output_dir: str = "figures/needle_in_a_haystack"

    @classmethod
    def from_global_config(cls, global_config: GlobalConfig):
        return cls(**{f.name: getattr(global_config, f.name) for f in fields(cls)})


def check_indices(
    indices: List[int],
    L: int,
):
    """
    Do some basic input checking of indices into the categorical Jacobian.
    """
    assert len(indices) == len(set(indices)), "Indices are not unique."
    assert max(indices) < L, "Some indices exceed the sequence length."
    assert min(indices) >= 0, "Only non-negative indices are supported."


def generate_random_dna_sequence(length: int, seed: int = 1337) -> str:
    """Generate a random sequence of nucleotides."""
    nucleotides = ["A", "C", "G", "T"]
    with temporary_seed(seed):
        return "".join(random.choice(nucleotides) for _ in range(length))


def generate_power_of_two_range(min_length: int, max_length: int) -> list[int]:
    """
    Generate a list that starts at `min_length` and ends at `max_length`, inclusive,
    with all powers of two in between.
    """
    start = math.ceil(math.log2(min_length))
    end = math.floor(math.log2(max_length))
    powers = [2**i for i in range(start, end + 1)]
    return (
        [min_length] + powers[1:] + ([max_length] if (len(powers) > 0 and max_length != powers[-1]) else [])
    )


def repeat_sequence_until(seq: str, min_length: int) -> str:
    """
    If a sequence is shorter than a `min_length`, repeat the string until it is
    longer than that length and trim off any extra characters.
    """
    return (seq * ((min_length - 1) // len(seq) + 1))[:min_length]


def do_apc(
    x: np.ndarray,
    rm: int = 1,
) -> np.ndarray:
    """
    Implements average product correction (APC).

    rm=0 remove none
    rm=1 apc
    """
    x = np.copy(x)

    if rm == 0:
        return x

    elif rm == 1:
        a1 = x.sum(0, keepdims=True)
        a2 = x.sum(1, keepdims=True)
        y = x - (a1 * a2) / x.sum()

    else:
        # Decompose matrix, remove largest eigenvector(s).
        u, s, v = np.linalg.svd(x)
        y = s[rm:] * u[:, rm:] @ v[rm:, :]

    np.fill_diagonal(y, 0)

    return y


def get_contacts(
    jac: np.ndarray,
    symm: bool = True,
    center: bool = True,
    full_jacobian: bool = True,
    rm: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Convert the Jacobian `jac` with shape (L_perturb, V, L_measure, V) to a contact map
    with shape (L_perturb, L_measure).

    A full Jacobian will have L_perturb == L_measure == L, where L is the sequence length.
    """
    assert len(jac.shape) == 4, "Jacobian must have 4 dimensions."

    if full_jacobian and jac.shape[0] != jac.shape[2]:
        warnings.warn(
            "get_contacts() expected full Jacobian but it is not square, will treat it as "
            "a partial Jacobian."
        )
        full_jacobian = False

    j = jac.copy()

    if center:
        for i in range(4):
            j -= j.mean(i, keepdims=True)

    j_fn = np.sqrt(np.square(j).sum((1, 3)))

    if full_jacobian:
        np.fill_diagonal(j_fn, 0)

        j_fn = do_apc(j_fn, rm=rm)

        if symm:
            j_fn = (j_fn + j_fn.T) / 2.0
    else:
        if verbose:
            warnings.warn(
                "Diagonal fill, APC, and symmetrization are not supported for " "partial Jacobians."
            )

    return j_fn


def make_figure(df: pd.DataFrame, needle_length: int, output_dir: str, model_name: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df.pivot(index="depth", columns="haystack_length", values="score")

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df,
        cmap="RdYlGn",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Score"},
        vmin=0.0,
        vmax=0.8,
    )
    plt.title(f"Needle (length {needle_length}) in a haystack")
    plt.xlabel("Haystack length")
    plt.ylabel("Depth into context (%)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/results_{model_name}.svg")
    plt.close()