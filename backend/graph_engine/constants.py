"""
Constants shared across the graph execution runtime.
"""


class NodePort:
    """
    Port index constants for node connections.

    Port indices are shared between the executor and validator to ensure
    both sides agree on the semantics of each connection.
    """

    DEFAULT_OUTPUT = 0
    IMAGE_INPUT = 0
    PROMPT_INPUT = 1
