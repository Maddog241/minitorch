from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from collections import deque, defaultdict

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_1 = vals[:arg] + (vals[arg]+epsilon, ) + vals[arg+1:]
    vals_2 = vals[:arg] + (vals[arg]-epsilon, ) + vals[arg+1:]
    return (f(*vals_1) - f(*vals_2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    def dfs(var):
        visited.add(var.unique_id)
        for input in var.history.inputs:
            if input.unique_id not in visited:
                dfs(input)
        variables.append(var)
    variables = []
    visited = set()
    dfs(variable)

    return variables[::-1]

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    id2deriv = {variable.unique_id: deriv}
    variables = topological_sort(variable=variable)
    for var in variables:
        assert var.unique_id in id2deriv
        var_deriv = id2deriv[var.unique_id]
        if not var.is_leaf():
            back = var.chain_rule(var_deriv)
            for prev_var, deriv in back:
                if prev_var.unique_id not in id2deriv:
                    id2deriv[prev_var.unique_id] = deriv
                else:
                    id2deriv[prev_var.unique_id] += deriv
        else:
            var.accumulate_derivative(var_deriv)

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
