"""Various Enums used across fimjax for options."""

from enum import StrEnum

class CHECKPOINT_NAMES(StrEnum):
    """Names of named checkpoints that can be used to specify what should be checkpointed."""
    UPDATE_TRIANGLES = 'update_triangles'


class ITERATION_SCHEME(StrEnum):
    """Iteration schemes that can be used for the solver function in FIM."""
    FOR = 'for'
    WHILE = 'while'
    WHILE_CHECKPOINTED = 'while_checkpointed'
    FOR_FIXED_POINT = 'for_fixedpoint'