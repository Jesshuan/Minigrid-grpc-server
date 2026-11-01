"""Acting gRPC client for the Minigrid API — terminal only, no window.

Run from the repository root:
    python -m grpc_api.act_client
    python -m grpc_api.act_client --debug
        (--debug prints the full observation payload received per response)
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


def _describe_obs(obs_decoded: dict) -> str:
    """Rebuild the numpy obs and describe it (see obs_codec.unpack_obs)."""
    
    image = obs_decoded["image"]
    line = (
        f"obs: numpy array shape={image.shape} dtype={image.dtype} "
        f"values={image.min()}..{image.max()}"
    )
    mental = obs_decoded["mental_map"]
    if mental is not None:
        line += (
            f" | mental map: {mental.shape[0]}x{mental.shape[1]} grayscale "
            f"values={mental.min()}..{mental.max()}"
        )
    return line


def _print_payload(obs_decoded: dict) -> None:
    """Print the full unpacked observation payload received over the wire.

    Debug mode only: the image array in full, plus direction/mission, and
    the mental map as an NxN grid of grayscale values (255 = agent,
    100 = goal).
    """
    image = obs_decoded["image"]
    print(f"  image: shape={image.shape} dtype={image.dtype}")
    print("  image values:")
    for line in str(image).splitlines():
        print(f"    {line}")
    if obs_decoded["direction"] is not None:
        print(f"  direction: {obs_decoded['direction']}")
    if obs_decoded["mission"]:
        print(f"  mission: {obs_decoded['mission']!r}")
    mental = obs_decoded["mental_map"]
    if mental is not None:
        print(f"  mental map ({mental.shape[0]}x{mental.shape[1]}):")
        for row in mental:
            print("    " + " ".join(f"{v:3d}" for v in row))


def run(args: argparse.Namespace) -> int:
    """Act over gRPC from the terminal; print the responses, no window."""
    if not sys.stdin.isatty():
        print("act mode needs an interactive terminal", file=sys.stderr)
        return 1
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = pb2_grpc.MinigridEnvStub(channel)

    print(KEYS_HELP)
    debug = args.debug

    try:

        response = stub.Reset(pb2.ResetRequest())
        obs_decoded = unpack_obs(response.observation)
        print(f"reset: {_describe_obs(obs_decoded)}")
        if debug:
            _print_payload(obs_decoded)

        while True:
            key = _read_key()
            if key in ("q", "Q", "\x03", "\x04"):  # q, Ctrl-C, Ctrl-D
                break
            if key == "r":
                response = stub.Reset(pb2.ResetRequest())
                obs_decoded = unpack_obs(response.observation)
                print(f"reset! {_describe_obs(obs_decoded)}")
                if debug:
                    _print_payload(obs_decoded)
                continue
            action = KEY_TO_ACTION.get(key)
            if action is None:
                continue
            response = stub.Step(pb2.StepRequest(action=action))
            obs_decoded = unpack_obs(response.observation)
            print(f"reward={response.reward:+.2f} terminated={response.terminated} {_describe_obs(obs_decoded)}")
            if debug:
                print('observation payload:')
                _print_payload(obs_decoded)

    except grpc.RpcError as exc:
        print(f"server error: {exc.code()} {exc.details()}", file=sys.stderr)
    finally:
        channel.close()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="server address")
    parser.add_argument("--port", type=int, default=50051, help="server port")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print the full observation payload received per response",
    )
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
