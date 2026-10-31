"""Encode/decode minigrid observations for the Minigrid gRPC API.

The wire format carries the observation's image as flat bytes; this module
is the single place that knows how to pack it server-side and rebuild the
numpy array client-side, so consumers never touch raw bytes themselves.
"""

from __future__ import annotations

import numpy as np

from grpc_api import minigrid_api_pb2 as pb2


def pack_obs(obs) -> pb2.Observation:
    """Serialize minigrid's obs into an Observation message.

    The default MiniGridEnv obs is the dict
    ``{"image": HxWx3 uint8, "direction": int, "mission": str}``; with
    ImgObsWrapper (--agent-view) the obs is the image ndarray alone, and
    direction/mission are left unset.
    """
    if isinstance(obs, dict):
        image, direction, mission = obs["image"], int(obs["direction"]), obs["mission"]
    else:
        image, direction, mission = obs, None, ""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    return pb2.Observation(
        image=image.tobytes(),
        height=height,
        width=width,
        direction=direction,
        mission=mission,
    )


def unpack_obs(obs: pb2.Observation) -> dict:
    """Rebuild a numpy observation from an Observation message.

    Returns minigrid's obs shape again: ``{"image": HxWx3 uint8 ndarray,
    "direction": int | None, "mission": str}`` — ready to feed a policy.
    """
    expected = obs.height * obs.width * 3
    if len(obs.image) != expected:
        raise ValueError(
            f"corrupt Observation: {len(obs.image)}B pixel data does not match "
            f"{obs.width}x{obs.height}x3"
        )
    image = np.frombuffer(obs.image, dtype=np.uint8).reshape(obs.height, obs.width, 3)
    return {
        "image": image,
        "direction": obs.direction if obs.HasField("direction") else None,
        "mission": obs.mission,
    }
