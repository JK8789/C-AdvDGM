import typing
from abc import abstractmethod
from typing import Any, Callable, Dict

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from mlc.constraints.relation_constraint import (
    AndConstraint,
    BaseRelationConstraint,
    Constant,
    ConstraintsNode,
    EqualConstraint,
    Feature,
    LessConstraint,
    LessEqualConstraint,
    MathOperation,
    OrConstraint,
)
from mlc.constraints.utils import get_feature_index

EPS: npt.NDArray[Any] = np.array(0.000001)


class ConstraintsVisitor:
    """Abstract Visitor Class"""

    @abstractmethod
    def visit(self, item: ConstraintsNode) -> Any:
        pass

    @abstractmethod
    def execute(self) -> Any:
        pass


class TensorFlowConstraintsVisitor(ConstraintsVisitor):
    def __init__(
        self,
        constraint: BaseRelationConstraint,
        x: "tf.Tensor",
        feature_names: npt.ArrayLike = None,
    ):
        self.constraint = constraint
        self.x = x
        self.feature_names = feature_names

    @staticmethod
    def get_zeros_tf(operands: typing.List["tf.Tensor"]) -> "tf.Tensor":
        import tensorflow as tf

        i = np.argmax([op.ndim for op in operands])
        return tf.zeros(operands[i].shape, dtype=operands[i].dtype)

    @staticmethod
    def str_operator_to_result_f() -> (
        Dict[str, Callable[["tf.Tensor", "tf.Tensor"], "tf.Tensor"]]
    ):
        import tensorflow as tf

        return {
            "+": lambda left, right: left + right,
            "-": lambda left, right: left - right,
            "*": lambda left, right: left * right,
            "/": lambda left, right: left / right,
            "**": lambda left, right: tf.math.pow(left, right),
        }

    def visit(self, constraint_node: ConstraintsNode) -> "tf.Tensor":
        import tensorflow as tf

        # ------------ Values
        if isinstance(constraint_node, Constant):
            return tf.constant(constraint_node.constant, dtype=tf.float32)

        elif isinstance(constraint_node, Feature):
            feature_index = get_feature_index(
                self.feature_names, constraint_node.feature_id
            )
            return self.x[:, feature_index]

        elif isinstance(constraint_node, MathOperation):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            operator = constraint_node.operator
            if operator in self.str_operator_to_result_f():
                return self.str_operator_to_result_f()[operator](
                    left_operand, right_operand
                )
            else:
                raise NotImplementedError

        # ------------ Constraints

        # ------ Binary
        elif isinstance(constraint_node, OrConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            local_min = operands[0]
            for i in range(1, len(operands)):
                local_min = tf.minimum(local_min, operands[i])
            return local_min

        elif isinstance(constraint_node, AndConstraint):
            operands = [e.accept(self) for e in constraint_node.operands]
            local_sum = operands[0]
            for i in range(1, len(operands)):
                local_sum = local_sum + operands[i]
            return local_sum

        # ------ Comparison
        elif isinstance(constraint_node, LessEqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            zeros = self.get_zeros_tf([left_operand, right_operand])
            return tf.maximum(zeros, (left_operand - right_operand))

        elif isinstance(constraint_node, LessConstraint):
            left_operand = constraint_node.left_operand.accept(self) + EPS
            right_operand = constraint_node.right_operand.accept(self)
            zeros = self.get_zeros_tf([left_operand, right_operand])
            return tf.maximum(zeros, (left_operand - right_operand))

        elif isinstance(constraint_node, EqualConstraint):
            left_operand = constraint_node.left_operand.accept(self)
            right_operand = constraint_node.right_operand.accept(self)
            return tf.abs(left_operand - right_operand)

        else:
            raise NotImplementedError

    def execute(self) -> "tf.Tensor":
        return self.constraint.accept(self)


class TensorFlowConstraintsExecutor:
    import tensorflow as tf

    def __init__(
        self,
        constraint: BaseRelationConstraint,
        feature_names: npt.ArrayLike = None,
    ):
        self.constraint = constraint
        self.feature_names = feature_names

    def execute(self, x: "tf.Tensor") -> "tf.Tensor":
        visitor = TensorFlowConstraintsVisitor(
            self.constraint, x, self.feature_names
        )
        return visitor.execute()
