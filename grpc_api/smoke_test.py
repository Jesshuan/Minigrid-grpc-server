"""Smoke test for the Minigrid gRPC API.

Starts its own server subprocess on port 50052, exercises Reset/Step/
StreamFrames/Status and error handling, prints PASS lines for each check.
Reset/Step responses carry the env's observation and scalars only — PNG
frames travel exclusively through StreamFrames.

Run from the repository root:
    python grpc_api/smoke_test.py
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time

import gymnasium as gym
import grpc
import minigrid  # noqa: F401  — registers all MiniGrid-* envs for gym.make
import numpy as np

from grpc_api import minigrid_api_pb2 as pb2, minigrid_api_pb2_grpc as pb2_grpc
from grpc_api.obs_codec import unpack_obs

PORT = "50052"
ENV_ID = "MiniGrid-LavaCrossingS9N3-v0"


def wait_for_server(proc: subprocess.Popen) -> pb2_grpc.MinigridEnvStub:
    channel = grpc.insecure_channel(f"127.0.0.1:{PORT}")
    stub = pb2_grpc.MinigridEnvStub(channel)
    for _ in range(100):
        time.sleep(0.1)
        if proc.poll() is not None:
            print(proc.stdout.read() if proc.stdout else "")
            sys.exit("server died before becoming ready")
        try:
            stub.Status(pb2.StatusRequest(), timeout=1)
            return stub
        except grpc.RpcError:
            pass
    sys.exit("server did not become ready")


def check_obs(obs: pb2.Observation, dict_mode: bool) -> dict:
    image = unpack_obs(obs)["image"]
    assert image.shape == (obs.height, obs.width, 3) and image.dtype == np.uint8
    assert obs.HasField("direction") == dict_mode, obs.HasField("direction")
    return image


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "grpc_api.server", "--env-id", ENV_ID, "--port", PORT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stub = wait_for_server(proc)

        status = stub.Status(pb2.StatusRequest())
        assert status.env_id == ENV_ID
        assert status.step_count == 0 and status.mission
        print("PASS status:", status.env_id, "| agent_pos:", status.agent_pos)

        reset = stub.Reset(pb2.ResetRequest(seed=42))
        image = check_obs(reset.observation, dict_mode=True)
        assert reset.step_count == 0 and reset.max_steps > 0
        # Bit-exact roundtrip: the decoded array must equal a locally seeded env.
        local_obs, _ = gym.make(ENV_ID).reset(seed=42)
        assert np.array_equal(image, local_obs["image"]), "obs roundtrip mismatch"
        print(
            f"PASS reset: obs shape={image.shape} dtype={image.dtype} "
            f"(bit-exact vs local env), mission={reset.observation.mission!r}"
        )

        last_step = reset.step_count
        for i in range(4):
            r = stub.Step(pb2.StepRequest(action=pb2.Action.FORWARD))
            assert r.step_count > last_step
            last_step = r.step_count
            check_obs(r.observation, dict_mode=True)
            assert not r.auto_reset and not r.terminated and not r.truncated
            print(f"PASS step {i + 1}: reward={r.reward:+.2f} step_count={r.step_count} obs={len(r.observation.image)}B")

        try:
            stub.Step(pb2.StepRequest(action=pb2.Action.ACTION_UNSPECIFIED))
            raise AssertionError("expected INVALID_ARGUMENT")
        except grpc.RpcError as e:
            assert e.code() == grpc.StatusCode.INVALID_ARGUMENT
            print("PASS unspecified action rejected:", e.code().name)

        # A step with no watcher connected must not render anything.
        assert stub.Status(pb2.StatusRequest()).last_frame_id == 0
        print("PASS lazy rendering: no frame generated without watchers")

        stream = stub.StreamFrames(pb2.StartStreamRequest())
        snap = next(stream)
        assert snap.kind == pb2.EVENT_SNAPSHOT
        print(f"PASS snapshot: {snap.width}x{snap.height} (PNG magic {snap.png[:4]!r})")
        stub.Step(pb2.StepRequest(action=pb2.Action.LEFT))
        live = next(stream)
        assert live.kind == pb2.EVENT_STEP, pb2.EventType.Name(live.kind)
        print("PASS live stream frame:", pb2.EventType.Name(live.kind), live.frame_id)
        stream.cancel()

        r1 = stub.Step(pb2.StepRequest(action=pb2.Action.RIGHT))
        r2 = stub.Step(pb2.StepRequest(action=pb2.Action.RIGHT))
        assert r1.step_count < r2.step_count
        print("PASS interleaved steps:", r1.step_count, "->", r2.step_count)
        print("ALL SMOKE CHECKS PASSED")
        return 0
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        if proc.stdout:
            print("--- server log tail ---")
            print(proc.stdout.read()[-1000:])


if __name__ == "__main__":
    sys.exit(main())
