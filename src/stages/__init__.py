"""Stage implementations for the debate workflow."""

from .role_assignment import RoleAssignment
from .solver import Solver

__all__ = [
    "RoleAssignment",
    "Solver",
]
