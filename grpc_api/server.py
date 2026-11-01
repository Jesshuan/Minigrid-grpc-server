"""gRPC server exposing a single Minigrid environment.

The environment is chosen at startup by its gymnasium registry id, exactly
like minigrid/manual_control.py does with --env-id. Acting clients call
Reset/Step and receive the env's observation and step scalars only — the
rendered frames travel exclusively through StreamFrames, and rendering is
skipped entirely while no watcher is subscribed.

Run from the repository root:
    python -m grpc_api.server --env-id MiniGrid-MultiRoom-N6-v0
    python -m grpc_api.server --env-id MiniGrid-MultiRoom-N6-v0 --debug
        (--debug prints the full observation payload of every response)
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
from concurrent import futures

import gymnasium as gym
import grpc
import numpy as np
from google.protobuf import empty_pb2 as google_empty_pb2

import minigrid  # noqa: F401  — registers all MiniGrid-*/BabyAI-* envs
from grpc_api import frames
from grpc_api import minigrid_api_pb2 as pb2
from grpc_api import minigrid_api_pb2_grpc as pb2_grpc
from grpc_api.obs_codec import pack_obs
from minigrid.core.actions import Actions
from minigrid.core.constants import DIR_TO_VEC, OBJECT_TO_IDX
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

logger = logging.getLogger("grpc_api.server")

# proto Action values are the Actions IntEnum values + 1 (proto3 reserves 0).
_ACTION_TO_ENUM = {
    Actions.left: pb2.Action.LEFT,
    Actions.right: pb2.Action.RIGHT,
    Actions.forward: pb2.Action.FORWARD,
    Actions.pickup: pb2.Action.PICKUP,
    Actions.drop: pb2.Action.DROP,
    Actions.toggle: pb2.Action.TOGGLE,
    Actions.done: pb2.Action.DONE,
}
_ENUM_TO_ACTION = {v: k for k, v in _ACTION_TO_ENUM.items()}

# A lagging watcher must never block the acting client: bounded per-watcher
# queues, drop-oldest on overflow.
_WATCHER_QUEUE_SIZE = 64

# Value written on the agent's cell in the mental map: the agent is not part
# of grid.encode (it lives outside the Grid), so it is overlaid distinctly.
AGENT_MAP_VALUE = 255
GOAL_MAP_VALUE = 100

def _goal_cell(unwrapped, world: np.ndarray) -> tuple[int, int] | None:
    """Locate the goal cell. Some envs (the crossings) and DirectionObsWrapper
    expose goal_position; plain envs (Empty, DoorKey...) do not — for those,
    look the goal object up in the encoded grid instead."""
    goal_position = getattr(unwrapped, "goal_position", None)
    if goal_position is not None:
        return goal_position
    hits = np.argwhere(world[:, :, 0] == OBJECT_TO_IDX["goal"])
    return None if hits.size == 0 else (int(hits[0][0]), int(hits[0][1]))


def wrap_mental_map(env) -> np.ndarray:
    """Aggregate the whole world grid into one grayscale NxN "mental map".

    N is the env's own grid size (width == height in MiniGrid), e.g. 9x9
    for LavaCrossingS9N3 — no resampling. Each cell is the mean of its
    three encoded channels (object id, color, state flags: door open/
    locked, ball inside a box, ...), so all cell information is aggregated
    into a single black&white value; ``AGENT_MAP_VALUE`` is written on the
    agent's cell since the agent is not part of grid.encode.

    ``env`` may be a wrapped env; the unwrapped one provides the world.
    """
    unwrapped = env.unwrapped
    world = unwrapped.grid.encode()  # (width, height, 3) uint8
    gray = np.round(world.mean(axis=2)).astype(np.uint8)
    # arrays are [x, y]-ordered (encode walks width first); agent_pos is (x, y)
    gray[unwrapped.agent_pos[0], unwrapped.agent_pos[1]] = AGENT_MAP_VALUE
    goal_cell = _goal_cell(unwrapped, world)
    if goal_cell is not None:
        gray[goal_cell[0], goal_cell[1]] = GOAL_MAP_VALUE
    return gray


def _debug_obs_payload(obs, mental_map: np.ndarray) -> None:
    """Dump the full observation payload being packed into a response.

    Called only in debug mode; the mental map is printed as an NxN grid,
    one aggregated cell value per world cell (255 = agent, 100 = goal).
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    image = obs["image"] if isinstance(obs, dict) else obs
    lines = [
        "observation payload:",
        f"  image: shape={image.shape} dtype={image.dtype}",
        "\n".join("    " + line for line in str(image).splitlines()),
    ]
    if isinstance(obs, dict):
        lines.append(f"  direction: {int(obs['direction'])}")
        lines.append(f"  mission: {obs['mission']!r}")
    lines.append(f"  mental map ({mental_map.shape[0]}x{mental_map.shape[1]}):")
    for row in mental_map:
        lines.append("    " + " ".join(f"{v:3d}" for v in row))
    logger.debug("%s", "\n".join(lines))


class FrameBroadcaster:
    """Fan-out of every generated Frame to zero or more StreamFrames subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[pb2.Frame]] = []
        self._broadcast_count = 0

    def register(self) -> queue.Queue[pb2.Frame]:
        q: queue.Queue[pb2.Frame] = queue.Queue(maxsize=_WATCHER_QUEUE_SIZE)
        with self._lock:
            self._subscribers.append(q)
            active = len(self._subscribers)
        logger.debug("watcher registered (%d active)", active)
        return q

    def unregister(self, q: queue.Queue[pb2.Frame]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def broadcast(self, frame: pb2.Frame) -> None:
        with self._lock:
            self._broadcast_count += 1
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(frame)
            except queue.Full:
                # Drop the oldest frame and keep the latest so a stalled
                # watcher resyncs to the live state instead of lagging.
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                except (queue.Empty, queue.Full):
                    pass

    @property
    def broadcast_count(self) -> int:
        with self._lock:
            return self._broadcast_count

    @property
    def active_watchers(self) -> int:
        with self._lock:
            return len(self._subscribers)


class EnvRunner:
    """Owns the single env; all mutation happens under one lock.

    Step semantics mirror ManualControl.step: on terminated/truncated the
    env is auto-reset (with the server's configured seed, like
    manual_control.reset(self.seed)) and the response reports the final
    step's reward/terminated/truncated together with the post-reset frame.
    """

    def __init__(self, env: gym.Env, env_id: str, seed: int | None, broadcaster: FrameBroadcaster) -> None:
        self.env = env
        self.env_id = env_id
        self.seed = seed
        self.broadcaster = broadcaster
        self._lock = threading.Lock()
        self._frame_count = 0
        self._closed = False

    def start(self) -> None:
        """Initial reset at startup, before the server starts serving.

        No frame is generated: nobody can be watching yet, and a watcher
        joining later receives a current-state snapshot.
        """
        with self._lock:
            self.env.reset(seed=self.seed)

    def _broadcast_frame(self, kind: int) -> None:
        """Render and fan out a frame — skipped while nobody is watching."""
        if self.broadcaster.active_watchers == 0:
            return
        png, width, height = frames.capture_frame(self.env)
        self._frame_count += 1
        self.broadcaster.broadcast(
            pb2.Frame(
                png=png,
                width=width,
                height=height,
                frame_id=self._frame_count,
                step_count=self.env.unwrapped.step_count,
                kind=kind,
            )
        )

    def _resolve_reset_seed(self, request_seed: int | None, has_request_seed: bool) -> int | None:
        if has_request_seed:
            self.seed = request_seed
            return request_seed
        # manual_control always resets with self.seed (the CLI --seed); with
        # no seed configured, continue the env's RNG stream.
        return self.seed

    def _mental_map(self) -> np.ndarray:
        """Whole symbolic world grid aggregated into a grayscale NxN array.

        Built straight from the unwrapped env's grid (no rendering), so it
        stays cheap: one aggregated cell value per world cell, agent
        overlaid with 255.
        """
        return wrap_mental_map(self.env)

    def do_reset(self, request: pb2.ResetRequest) -> pb2.ResetResponse:
        with self._lock:
            self._check_open()
            seed = self._resolve_reset_seed(
                request.seed, request.HasField("seed")
            )
            obs, _ = self.env.reset(seed=seed)
            self._broadcast_frame(pb2.EVENT_RESET)
            unwrapped: MiniGridEnv = self.env.unwrapped
            mental_map = self._mental_map()
            _debug_obs_payload(obs, mental_map)
            logger.info(
                "reset (seed=%s): step_count=%d/%d",
                seed, unwrapped.step_count, unwrapped.max_steps,
            )
            return pb2.ResetResponse(
                observation=pack_obs(obs, mental_map),
                step_count=unwrapped.step_count,
                max_steps=unwrapped.max_steps,
            )

    def do_step(self, request: pb2.StepRequest) -> pb2.StepResponse:
        with self._lock:
            self._check_open()
            action = _ENUM_TO_ACTION.get(request.action)
            if action is None:
                raise ValueError(f"unset or unknown action: {request.action!r}")
            unwrapped: MiniGridEnv = self.env.unwrapped
            obs, reward, terminated, truncated, _ = self.env.step(action)

            mental_map = self._mental_map()
            _debug_obs_payload(obs, mental_map)

            if terminated or truncated:
                # manual_control prints "terminated!"/"truncated!" then
                # self.reset(self.seed) — auto-reset with the configured seed.
                # The response keeps the FINAL step's data (its observation is
                # captured before the reset discards it).
                outcome = "terminated!" if terminated else "truncated!"
                logger.info(
                    "episode %s at step_count=%d/%d (final reward=%+.2f), auto-reset",
                    outcome, unwrapped.step_count, unwrapped.max_steps, reward,
                )
                final_step_count = unwrapped.step_count
                self.env.reset(seed=self.seed)
                self._broadcast_frame(pb2.EVENT_AUTO_RESET)
                logger.info(
                    "auto-reset done: step_count=%d/%d (max_steps unchanged)",
                    unwrapped.step_count, unwrapped.max_steps,
                )
                return pb2.StepResponse(
                    observation=pack_obs(obs, mental_map),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    auto_reset=True,
                    step_count=final_step_count,
                )

            logger.info(
                "step: step_count=%d/%d reward=%+.2f terminated=False truncated=False",
                unwrapped.step_count, unwrapped.max_steps, reward,
            )
            self._broadcast_frame(pb2.EVENT_STEP)
            return pb2.StepResponse(
                observation=pack_obs(obs, mental_map),
                reward=reward,
                terminated=False,
                truncated=False,
                auto_reset=False,
                step_count=unwrapped.step_count,
            )

    def snapshot_frame(self) -> pb2.Frame:
        """Current-state frame for a watcher joining mid-episode."""
        with self._lock:
            png, width, height = frames.capture_frame(self.env)
            return pb2.Frame(
                png=png,
                width=width,
                height=height,
                frame_id=self._frame_count,
                step_count=self.env.unwrapped.step_count,
                kind=pb2.EVENT_SNAPSHOT,
            )

    def get_status(self) -> pb2.StatusResponse:
        with self._lock:
            unwrapped: MiniGridEnv = self.env.unwrapped
            return pb2.StatusResponse(
                env_id=self.env_id,
                mission=unwrapped.mission,
                step_count=unwrapped.step_count,
                max_steps=unwrapped.max_steps,
                frames_broadcast=self.broadcaster.broadcast_count,
                active_watchers=self.broadcaster.active_watchers,
                last_frame_id=self._frame_count,
                agent_pos=f"({int(unwrapped.agent_pos[0])}, {int(unwrapped.agent_pos[1])}), facing {DIR_TO_VEC[unwrapped.agent_dir]}",
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.env.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("environment is closed")


class MinigridEnvServicer(pb2_grpc.MinigridEnvServicer):
    def __init__(self, runner: EnvRunner) -> None:
        self.runner = runner

    def Reset(self, request: pb2.ResetRequest, context: grpc.ServicerContext) -> pb2.ResetResponse:
        try:
            return self.runner.do_reset(request)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # noqa: BLE001 — never let handler threads die
            logger.exception("Reset failed")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def Step(self, request: pb2.StepRequest, context: grpc.ServicerContext) -> pb2.StepResponse:
        try:
            return self.runner.do_step(request)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Step failed")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def StreamFrames(self, request: pb2.StartStreamRequest, context: grpc.ServicerContext):
        broadcaster = self.runner.broadcaster
        q = broadcaster.register()
        try:
            # Snapshot so a watcher joining mid-episode has something to show
            # immediately; no history replay is kept.
            yield self.runner.snapshot_frame()
            while context.is_active():
                try:
                    yield q.get(timeout=1.0)  # periodic wake to notice cancellation
                except queue.Empty:
                    continue
        finally:
            broadcaster.unregister(q)

    def Status(self, request, context: grpc.ServicerContext) -> pb2.StatusResponse:
        return self.runner.get_status()

    def Close(self, request, context: grpc.ServicerContext):
        self.runner.close()
        return google_empty_pb2.Empty()


def build_env(args: argparse.Namespace) -> gym.Env:
    """Env construction identical to manual_control.py, with rgb_array rendering."""
    env = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="rgb_array",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
        **({"max_steps": args.max_steps} if args.max_steps is not None else {}),
    )
    if args.agent_view:
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)
    return env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-MultiRoom-N6-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="render what the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default=640,
        help="set the resolution for rendering (width and height)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override the episode step limit (passthrough to the env)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=50051, help="listen port")
    parser.add_argument("--max-workers", type=int, default=16, help="grpc thread pool size")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print the full observation payload of every reset/step response",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    env = build_env(args)
    broadcaster = FrameBroadcaster()
    runner = EnvRunner(env, env_id=args.env_id, seed=args.seed, broadcaster=broadcaster)
    runner.start()  # initial reset + first frame before serving

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers),
        options=[
            ("grpc.max_receive_message_length", 1 * 1024 * 1024),
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
        ],
    )
    pb2_grpc.add_MinigridEnvServicer_to_server(MinigridEnvServicer(runner), server)
    bind_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(bind_addr)
    server.start()
    logger.info(
        "minigrid gRPC serving %s at %s (seed=%s, agent_view=%s)",
        args.env_id,
        bind_addr,
        args.seed,
        args.agent_view,
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("shutting down")
        server.stop(grace=1.0).wait()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
