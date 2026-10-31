"""Acting gRPC client for the Minigrid API — terminal only, no window.

Sends single keypresses to the server and prints the responses
(observation, reward, terminated/truncated). Rendering is not its job:
the passive viewer lives in watch_client.py, and frames travel only on
StreamFrames.

Run from the repository root:
    python -m grpc_api.act_client
"""

from __future__ import annotations

import argparse
import logging
import sys

import grpc

from grpc_api import minigrid_api_pb2 as pb2
from grpc_api import minigrid_api_pb2_grpc as pb2_grpc
from grpc_api.obs_codec import unpack_obs

logger = logging.getLogger("grpc_api.act_client")

# Terminal keys, mirroring manual_control's physical controls.
KEY_TO_ACTION: dict[str, int] = {
    "a": pb2.Action.LEFT,
    "d": pb2.Action.RIGHT,
    "w": pb2.Action.FORWARD,
    " ": pb2.Action.TOGGLE,
    "p": pb2.Action.PICKUP,
    "o": pb2.Action.DROP,
    "e": pb2.Action.DONE,
    # alias of the older manual_control binding, for muscle memory
    "\r": pb2.Action.DONE,
}

KEYS_HELP = """\
act client keys:  a/d/w = left/right/forward   space = toggle
                  p = pickup   o = drop        e/return = done
                  r = reset    q = quit"""


def _read_key() -> str:
    """Read one keypress from the terminal in raw mode."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _describe_obs(obs: pb2.Observation) -> str:
    """Rebuild the numpy obs and describe it (see obs_codec.unpack_obs)."""
    image = unpack_obs(obs)["image"]
    print(image)
    kind = "agent-view image" if not obs.HasField("direction") else "dict"
    return (
        f"obs: numpy array shape={image.shape} dtype={image.dtype} "
        f"values={image.min()}..{image.max()} ({kind})"
    )


def run(args: argparse.Namespace) -> int:
    """Act over gRPC from the terminal; print the responses, no window."""
    if not sys.stdin.isatty():
        print("act mode needs an interactive terminal", file=sys.stderr)
        return 1
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = pb2_grpc.MinigridEnvStub(channel)

    print(KEYS_HELP)
    try:
        response = stub.Reset(pb2.ResetRequest())
        obs = response.observation
        print(f"reset: {_describe_obs(obs)} step={response.step_count} max_steps={response.max_steps}")
        while True:
            key = _read_key()
            if key in ("q", "Q", "\x03", "\x04"):  # q, Ctrl-C, Ctrl-D
                break
            if key == "r":
                response = stub.Reset(pb2.ResetRequest())
                print(f"reset!\n{response.observation.mission}")
                continue
            action = KEY_TO_ACTION.get(key)
            if action is None:
                continue
            response = stub.Step(pb2.StepRequest(action=action))
            obs = response.observation
            line = (
                f"step={response.step_count} reward={response.reward:+.2f} "
                f"{_describe_obs(obs)}"
            )
            if response.terminated or response.truncated:
                line += "  terminated!" if response.terminated else "  truncated!"
            if response.auto_reset:
                line += " (auto-reset)"
            print(line)
            if obs.mission:
                print(obs.mission)
    except grpc.RpcError as exc:
        print(f"server error: {exc.code()} {exc.details()}", file=sys.stderr)
    finally:
        channel.close()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="server address")
    parser.add_argument("--port", type=int, default=50051, help="server port")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    try:
        return run(args)
    except grpc.RpcError as exc:
        print(f"cannot reach server at {args.host}:{args.port}: {exc.code()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
