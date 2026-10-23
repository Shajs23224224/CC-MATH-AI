"""Bounded expression-tree search primitives; no candidate source code is executed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from core.contracts.adaptive_math import ExpressionNode

FloatArray: TypeAlias = NDArray[np.float64]
_BINARY = ("add", "sub", "mul", "div")
_UNARY = ("sin", "cos", "tanh", "abs")


@dataclass(frozen=True, slots=True)
class Node:
    """Internal immutable genome using an allow-listed expression grammar."""

    op: str
    feature_index: int | None = None
    constant: float | None = None
    children: tuple[Node, ...] = ()


class InvalidExpression(ValueError):
    """An expression violates its domain or numerical execution budget."""


def node_count(node: Node) -> int:
    return 1 + sum(node_count(child) for child in node.children)


def node_depth(node: Node) -> int:
    return 0 if not node.children else 1 + max(node_depth(child) for child in node.children)


def terminal(rng: np.random.Generator, feature_count: int) -> Node:
    if rng.random() < 0.72:
        return Node("feature", feature_index=int(rng.integers(0, feature_count)))
    value = float(np.round(rng.uniform(-3.0, 3.0), 3))
    return Node("constant", constant=value)


def random_tree(
    rng: np.random.Generator,
    feature_count: int,
    max_depth: int,
    max_nodes: int,
) -> Node:
    """Generate a genuine random expression within fixed depth and node budgets."""

    def build(depth: int, budget: int) -> Node:
        if depth <= 0 or budget <= 1 or rng.random() < 0.36:
            return terminal(rng, feature_count)
        if budget < 3 or rng.random() < 0.30:
            op = str(rng.choice(_UNARY))
            return Node(op, children=(build(depth - 1, budget - 1),))
        op = str(rng.choice(_BINARY))
        left_budget = int(rng.integers(1, budget - 1))
        right_budget = budget - 1 - left_budget
        return Node(
            op,
            children=(
                build(depth - 1, left_budget),
                build(depth - 1, right_budget),
            ),
        )

    result = build(max_depth, max_nodes)
    if node_count(result) > max_nodes or node_depth(result) > max_depth:
        return terminal(rng, feature_count)
    return result


def all_paths(node: Node, prefix: tuple[int, ...] = ()) -> tuple[tuple[int, ...], ...]:
    paths = [prefix]
    for index, child in enumerate(node.children):
        paths.extend(all_paths(child, prefix + (index,)))
    return tuple(paths)


def subtree_at(node: Node, path: tuple[int, ...]) -> Node:
    current = node
    for index in path:
        current = current.children[index]
    return current


def replace_subtree(node: Node, path: tuple[int, ...], replacement: Node) -> Node:
    if not path:
        return replacement
    index = path[0]
    children = list(node.children)
    children[index] = replace_subtree(children[index], path[1:], replacement)
    return Node(node.op, node.feature_index, node.constant, tuple(children))


def mutate_tree(
    node: Node,
    rng: np.random.Generator,
    feature_count: int,
    max_depth: int,
    max_nodes: int,
) -> Node:
    paths = all_paths(node)
    path = paths[int(rng.integers(0, len(paths)))]
    previous = subtree_at(node, path)
    remaining_nodes = max_nodes - node_count(node) + node_count(previous)
    remaining_depth = max_depth - len(path)
    replacement = random_tree(
        rng,
        feature_count,
        max(0, remaining_depth),
        max(1, remaining_nodes),
    )
    candidate = replace_subtree(node, path, replacement)
    if node_count(candidate) <= max_nodes and node_depth(candidate) <= max_depth:
        return candidate
    return random_tree(rng, feature_count, max_depth, max_nodes)


def crossover_trees(
    left: Node,
    right: Node,
    rng: np.random.Generator,
    feature_count: int,
    max_depth: int,
    max_nodes: int,
) -> Node:
    left_path = all_paths(left)[int(rng.integers(0, len(all_paths(left))))]
    right_paths = all_paths(right)
    right_path = right_paths[int(rng.integers(0, len(right_paths)))]
    candidate = replace_subtree(left, left_path, subtree_at(right, right_path))
    if node_count(candidate) <= max_nodes and node_depth(candidate) <= max_depth:
        return candidate
    return mutate_tree(left, rng, feature_count, max_depth, max_nodes)


def to_contract(node: Node) -> ExpressionNode:
    return ExpressionNode(
        op=node.op,  # type: ignore[arg-type]
        feature_index=node.feature_index,
        constant=node.constant,
        children=tuple(to_contract(child) for child in node.children),
    )


def from_contract(node: ExpressionNode) -> Node:
    return Node(
        node.op,
        node.feature_index,
        node.constant,
        tuple(from_contract(child) for child in node.children),
    )


def render_tree(node: Node, feature_names: tuple[str, ...]) -> str:
    if node.op == "feature":
        assert node.feature_index is not None
        return feature_names[node.feature_index]
    if node.op == "constant":
        assert node.constant is not None
        return f"{node.constant:g}"
    if node.op in _UNARY:
        return f"{node.op}({render_tree(node.children[0], feature_names)})"
    symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    symbol = symbols[node.op]
    return (
        f"({render_tree(node.children[0], feature_names)} {symbol} "
        f"{render_tree(node.children[1], feature_names)})"
    )


def evaluate_tree(
    node: Node,
    matrix: FloatArray,
    *,
    max_depth: int = 16,
    max_nodes: int = 511,
) -> FloatArray:
    """Interpret only authorized operations, enforcing depth, size and finite-value bounds."""
    if node_count(node) > max_nodes or node_depth(node) > max_depth:
        raise InvalidExpression("expression exceeds its execution budget")
    rows, width = matrix.shape
    visited = 0

    def evaluate(current: Node) -> FloatArray:
        nonlocal visited
        visited += 1
        if visited > max_nodes:
            raise InvalidExpression("expression exceeds its node budget")
        if current.op == "feature":
            if current.feature_index is None or current.feature_index >= width:
                raise InvalidExpression("expression refers to a missing feature")
            return matrix[:, current.feature_index]
        if current.op == "constant":
            if current.constant is None or not np.isfinite(current.constant):
                raise InvalidExpression("expression has an invalid constant")
            return np.full(rows, current.constant, dtype=np.float64)
        values = [evaluate(child) for child in current.children]
        with np.errstate(all="ignore"):
            if current.op == "add":
                result = values[0] + values[1]
            elif current.op == "sub":
                result = values[0] - values[1]
            elif current.op == "mul":
                result = values[0] * values[1]
            elif current.op == "div":
                if np.any(np.abs(values[1]) <= 1e-10):
                    raise InvalidExpression("division denominator is too close to zero")
                result = values[0] / values[1]
            elif current.op == "sin":
                result = np.sin(values[0])
            elif current.op == "cos":
                result = np.cos(values[0])
            elif current.op == "tanh":
                result = np.tanh(values[0])
            elif current.op == "abs":
                result = np.abs(values[0])
            else:
                raise InvalidExpression("operation is not allow-listed")
        if not np.all(np.isfinite(result)) or np.any(np.abs(result) > 1e12):
            raise InvalidExpression("expression produced a non-finite or excessive value")
        return np.asarray(result, dtype=np.float64)

    return evaluate(node)
