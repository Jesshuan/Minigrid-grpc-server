"""gRPC API for Minigrid.

Lets another microservice or a human act on and observe a single Minigrid
environment over the network. The environment is chosen at server startup by
its gymnasium registry id, exactly like ``minigrid/manual_control.py`` does
with ``--env-id``.

Run the server from the repository root:
    python -m grpc_api.server
Run the reference clients:
    python -m grpc_api.act_client     # act (terminal only, no window)
    python -m grpc_api.watch_client   # observe (pygame window)
"""

__version__ = "0.1.0"
