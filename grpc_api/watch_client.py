"""Passive-viewer gRPC client for the Minigrid API — pygame window, no acting.

Displays the server's StreamFrames output (current-state snapshot, then
every frame generated: resets, steps, auto-resets), like manual_control.py's
window without the keyboard. Frames are rendered only while at least one
viewer is subscribed, so the acting client is never burdened by rendering
for nobody.

Run from the repository root:
    python -m grpc_api.watch_client
    python -m grpc_api.watch_client --save-frames /tmp/f   # dump PNGs too
"""

from __future__ import annotations

import argparse
import io
import logging
import queue
import sys
import threading

import grpc
import pygame

from grpc_api import minigrid_api_pb2 as pb2
from grpc_api import minigrid_api_pb2_grpc as pb2_grpc

logger = logging.getLogger("grpc_api.watch_client")

WINDOW_SIZE = 640


class FrameDisplay:
    """Pygame window blitting PNG frames received from the server."""

    def __init__(self, window_size: int, save_dir: str | None = None) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Minigrid gRPC watch client")
        self.save_dir = save_dir
        self._saved = 0
        self._clock = pygame.time.Clock()

    def blit(self, png_bytes: bytes, caption: str = "") -> bool:
        """Display a PNG frame; returns False if the window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        try:
            image = pygame.image.load(io.BytesIO(png_bytes))
        except Exception:
            logger.exception("bad frame")
            return True
        scaled = pygame.transform.smoothscale(
            image,
            (
                WINDOW_SIZE,
                int(WINDOW_SIZE * image.get_height() / image.get_width()),
            ),
        )
        self.screen.fill((20, 20, 20))
        self.screen.blit(scaled, (0, 0))
        pygame.display.set_caption(f"Minigrid gRPC watch client  {caption}")
        pygame.display.flip()
        if self.save_dir:
            self._saved += 1
            (path := f"{self.save_dir}/frame_{self._saved:05d}.png")
            with open(path, "wb") as f:
                f.write(png_bytes)
        return True

    def tick(self, fps: int = 30) -> None:
        self._clock.tick(fps)

    def close(self) -> None:
        pygame.quit()


def _stream_frames(stub: pb2_grpc.MinigridEnvStub, frames_q: queue.Queue, done: threading.Event) -> None:
    try:
        for frame in stub.StreamFrames(pb2.StartStreamRequest()):
            if done.is_set():
                break
            try:
                frames_q.put_nowait(frame)
            except queue.Full:
                try:
                    frames_q.get_nowait()
                    frames_q.put_nowait(frame)
                except (queue.Empty, queue.Full):
                    pass
    except grpc.RpcError as exc:
        if exc.code() != grpc.StatusCode.CANCELLED:
            logger.error("StreamFrames failed: %s", exc)
    finally:
        done.set()


def run(args: argparse.Namespace) -> int:
    """Observe only: display whatever the server broadcasts."""
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = pb2_grpc.MinigridEnvStub(channel)
    display = FrameDisplay(WINDOW_SIZE, save_dir=args.save_frames)
    frames_q: queue.Queue = queue.Queue(maxsize=4)
    done = threading.Event()
    stream_thread = threading.Thread(target=_stream_frames, args=(stub, frames_q, done), daemon=True)
    stream_thread.start()
    try:
        while not done.is_set() or not frames_q.empty():
            try:
                frame = frames_q.get(timeout=0.5)
            except queue.Empty:
                # Keep the window responsive while no frame arrives.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done.set()
                display.tick()
                continue
            caption = f"step={frame.step_count} {pb2.EventType.Name(frame.kind)}"
            if not display.blit(frame.png, caption):
                break
            display.tick()
    finally:
        done.set()
        channel.close()
        display.close()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="server address")
    parser.add_argument("--port", type=int, default=50051, help="server port")
    parser.add_argument(
        "--save-frames",
        type=str,
        default=None,
        help="save every displayed frame as PNG files in this directory "
        "(useful to verify headless, without a display)",
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
