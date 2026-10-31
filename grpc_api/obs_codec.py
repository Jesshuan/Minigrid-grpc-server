"""Encode/decode minigrid observations for the Minigrid gRPC API.

The wire format carries the observation's image as flat bytes; this module
is the single place that knows how to pack it server-side and rebuild the
numpy array client-side, so consumers never touch raw bytes themselves.
"""

from __future__ import annotations

import numpy as np

from grpc_api import minigrid_api_pb2 as pb2



def pack_obs(obs, mental_map: np.ndarray | None = None) -> pb2.Observation:
    """Serialize minigrid's obs into an Observation message.

    The default MiniGridEnv obs is the dict
    ``{"image": HxWx3 uint8, "direction": int, "mission": str}``; with
    ImgObsWrapper (--agent-view) the obs is the image ndarray alone, and
    direction/mission are left unset. When ``mental_map`` is given (see
    wrap_mental_map) it is packed as the square grayscale map.
    """
    if isinstance(obs, dict):
        image, direction, mission = obs["image"], int(obs["direction"]), obs["mission"]
    else:
        image, direction, mission = obs, None, ""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    message = pb2.Observation(
        image=image.tobytes(),
        height=height,
        width=width,
        direction=direction,
        mission=mission,
    )
    if mental_map is not None:
        gray = np.ascontiguousarray(mental_map, dtype=np.uint8)
        if gray.ndim != 2 or gray.shape[0] != gray.shape[1]:
            raise ValueError(f"mental map must be a square 2D array, got shape {gray.shape}")
        message.mental_map = gray.tobytes()
        message.mental_map_size = gray.shape[0]
    return message


def unpack_obs(obs: pb2.Observation) -> dict:
    """Rebuild a numpy observation from an Observation message.

    Returns minigrid's obs shape again: ``{"image": HxWx3 uint8 ndarray,
    "direction": int | None, "mission": str}`` — ready to feed a policy.
    When the server packed one, the dict also carries ``"mental_map"``:
    an NxN uint8 grayscale ndarray (else None).
    """
    expected = obs.height * obs.width * 3
    if len(obs.image) != expected:
        raise ValueError(
            f"corrupt Observation: {len(obs.image)}B pixel data does not match "
            f"{obs.width}x{obs.height}x3"
        )
    image = np.frombuffer(obs.image, dtype=np.uint8).reshape(obs.height, obs.width, 3)
    mental_map = None
    if obs.mental_map_size and obs.mental_map:
        map_expected = obs.mental_map_size**2
        if len(obs.mental_map) != map_expected:
            raise ValueError(
                f"corrupt Observation: {len(obs.mental_map)}B mental map does not match "
                f"{obs.mental_map_size}x{obs.mental_map_size}"
            )
        mental_map = np.frombuffer(obs.mental_map, dtype=np.uint8).reshape(
            obs.mental_map_size, obs.mental_map_size
        )
    return {
        "image": image,
        "direction": obs.direction if obs.HasField("direction") else None,
        "mission": obs.mission,
        "mental_map": mental_map,
    }
