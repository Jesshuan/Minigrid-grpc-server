# Minigrid gRPC API

A gRPC service that lets **another microservice or a human** act on and
observe a Minigrid environment over the network. The environment is chosen
at server startup by its gymnasium registry id — exactly like
`minigrid/manual_control.py` does with `--env-id`.

The two concerns are kept apart:

- **Acting** (`Reset`/`Step`) returns the env's own `observation` plus the
  step scalars (`reward`, `terminated`, `truncated`, ...) — never a PNG.
- **Watching** (`StreamFrames`) is the only path that carries rendered
  frames, and the server skips rendering entirely while nobody watches.

## Layout

| File | Purpose |
|---|---|
| `minigrid_api.proto` | API definition (source of truth) |
| `minigrid_api_pb2*.py` | generated stubs (committed; consumers only need `grpcio`) |
| `frames.py` | render → PNG capture (applies the display transpose) |
| `obs_codec.py` | observation serialization: `pack_obs` (server) / `unpack_obs` (numpy rebuild, client) |
| `server.py` | env runner, frame broadcaster, gRPC servicer, argparse |
| `act_client.py` | acting client: terminal-only keypresses → Step/Reset, prints responses |
| `watch_client.py` | passive viewer: pygame window on the `StreamFrames` stream |

## Install

In your environment (with minigrid installed):

```bash
pip install -e ".[grpc]"     # from the repo root
# or: pip install grpcio grpcio-tools protobuf Pillow
```

## Running

From the repository root:

```bash
# Server (headless-safe; no pygame window needed)
python -m grpc_api.server --env-id MiniGrid-MultiRoom-N6-v0

# Acting client: terminal-only, prints observation/reward/terminated, no window
python -m grpc_api.act_client

# Second terminal: passive viewer of the same episode (pygame window)
python -m grpc_api.watch_client
```

Server flags mirror `manual_control.py` (`--env-id`, `--seed`, `--tile-size`,
`--agent-view`, `--agent-view-size`, `--screen-size`) plus `--host` (default
`0.0.0.0`), `--port` (default `50051`), `--max-workers`, and `--max-steps`
(episode step-limit passthrough, handy for tests).

### Keys (act_client)

Terminal-only (raw-mode stdin): **a/d/w** left/right/forward, **space**
toggle, **p** pickup, **o** drop, **e/return** done, **r** reset, **q**
quit. `watch_client` only displays frames (close its window to leave).

## API

```
service MinigridEnv {
  rpc Reset(ResetRequest) returns (ResetResponse);       // reset -> observation
  rpc Step(StepRequest) returns (StepResponse);          // act -> observation + reward/…
  rpc StreamFrames(StartStreamRequest) returns (stream Frame);  // passive viewers
  rpc Status(StatusRequest) returns (StatusResponse);    // introspection, no frame
  rpc Close(google.protobuf.Empty) returns (google.protobuf.Empty);
}
```

- `Observation` flattens minigrid's official obs: `image` (HxWx3 uint8,
  flat bytes on the wire — `bytes` values are small ints: object type, color,
  state), `direction` and `mission` in the default dict-obs mode. With
  `--agent-view` (`ImgObsWrapper`) the obs is the image alone, so
  `direction`/`mission` are unset. **Don't decode by hand** — use
  `grpc_api.obs_codec.unpack_obs` to rebuild the numpy array:

  ```python
  import grpc
  from grpc_api import minigrid_api_pb2 as pb2, minigrid_api_pb2_grpc as pb2_grpc
  from grpc_api.obs_codec import unpack_obs

  channel = grpc.insecure_channel("localhost:50051")
  stub = pb2_grpc.MinigridEnvStub(channel)
  resp = stub.Step(pb2.StepRequest(action=pb2.Action.FORWARD))
  obs = unpack_obs(resp.observation)   # {"image": HxWx3 uint8 ndarray, ...}
  print(resp.reward, resp.step_count, obs["image"].shape, obs["direction"])
  ```
- `Frame` (PNG + dimensions + monotonically increasing `frame_id`) travels
  only on `StreamFrames`. Rendering happens only while at least one watcher
  is subscribed.
- `StreamFrames` sends a current-state snapshot on connect, then every frame
  generated afterwards. No history replay. Slow watchers get a bounded
  queue with drop-oldest, so they can never block the acting client.

## Semantics worth knowing

- **One env per server process**, chosen at startup. Any number of clients
  may call `Step`; their actions interleave under a server-side lock (no
  turn-taking/fairness guarantee — coordinate externally if that matters).
- **Auto-reset** mirrors `ManualControl.step`: when an episode ends, the
  server resets immediately. In an `auto_reset=true` `StepResponse`, all
  fields (`observation`, `reward`, `terminated`, `truncated`, `step_count`)
  describe the *final step* of the ended episode; the post-reset state is
  observable on `StreamFrames` (`EVENT_AUTO_RESET`).
- **Seeds**: the server keeps the `--seed` given at startup and reuses it on
  every auto-reset, like `manual_control.reset(self.seed)`. `ResetRequest`
  with `seed` set overrides it (and updates the server's seed). Unset +
  server started without `--seed` → continues the RNG stream.
- **Mission** on `Status` is always read from `env.unwrapped.mission`
  (reliable even with the `--agent-view` wrapper stack, where observations
  are bare images).
- **`Action` enum offset**: proto3 requires the first enum value to be 0, so
  proto value = `minigrid.core.actions.Actions` value + 1 (see the proto
  comments; the server does the mapping).

## Regenerating stubs

After editing `minigrid_api.proto`:

```bash
python -m grpc_tools.protoc -I grpc_api \
  --python_out=grpc_api --grpc_python_out=grpc_api --pyi_out=grpc_api \
  grpc_api/minigrid_api.proto
```

Then reapply the manual relative-import shim at the top of
`minigrid_api_pb2_grpc.py` (grpcio-tools emits an absolute
`import minigrid_api_pb2`, which breaks `python -m grpc_api.*` — see the
`try: from grpc_api import ...` block marked "manual edit").
