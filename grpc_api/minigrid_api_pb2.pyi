from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Action(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_UNSPECIFIED: _ClassVar[Action]
    LEFT: _ClassVar[Action]
    RIGHT: _ClassVar[Action]
    FORWARD: _ClassVar[Action]
    PICKUP: _ClassVar[Action]
    DROP: _ClassVar[Action]
    TOGGLE: _ClassVar[Action]
    DONE: _ClassVar[Action]

class EventType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EVENT_TYPE_UNSPECIFIED: _ClassVar[EventType]
    EVENT_RESET: _ClassVar[EventType]
    EVENT_STEP: _ClassVar[EventType]
    EVENT_AUTO_RESET: _ClassVar[EventType]
    EVENT_SNAPSHOT: _ClassVar[EventType]
ACTION_UNSPECIFIED: Action
LEFT: Action
RIGHT: Action
FORWARD: Action
PICKUP: Action
DROP: Action
TOGGLE: Action
DONE: Action
EVENT_TYPE_UNSPECIFIED: EventType
EVENT_RESET: EventType
EVENT_STEP: EventType
EVENT_AUTO_RESET: EventType
EVENT_SNAPSHOT: EventType

class Frame(_message.Message):
    __slots__ = ("png", "width", "height", "frame_id", "step_count", "kind")
    PNG_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_COUNT_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    png: bytes
    width: int
    height: int
    frame_id: int
    step_count: int
    kind: EventType
    def __init__(self, png: _Optional[bytes] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., frame_id: _Optional[int] = ..., step_count: _Optional[int] = ..., kind: _Optional[_Union[EventType, str]] = ...) -> None: ...

class Observation(_message.Message):
    __slots__ = ("image", "height", "width", "direction", "mission")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    MISSION_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    height: int
    width: int
    direction: int
    mission: str
    def __init__(self, image: _Optional[bytes] = ..., height: _Optional[int] = ..., width: _Optional[int] = ..., direction: _Optional[int] = ..., mission: _Optional[str] = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("seed",)
    SEED_FIELD_NUMBER: _ClassVar[int]
    seed: int
    def __init__(self, seed: _Optional[int] = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ("observation", "step_count", "max_steps")
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    STEP_COUNT_FIELD_NUMBER: _ClassVar[int]
    MAX_STEPS_FIELD_NUMBER: _ClassVar[int]
    observation: Observation
    step_count: int
    max_steps: int
    def __init__(self, observation: _Optional[_Union[Observation, _Mapping]] = ..., step_count: _Optional[int] = ..., max_steps: _Optional[int] = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("action",)
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: Action
    def __init__(self, action: _Optional[_Union[Action, str]] = ...) -> None: ...

class StepResponse(_message.Message):
    __slots__ = ("observation", "reward", "terminated", "truncated", "auto_reset", "step_count")
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    AUTO_RESET_FIELD_NUMBER: _ClassVar[int]
    STEP_COUNT_FIELD_NUMBER: _ClassVar[int]
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    auto_reset: bool
    step_count: int
    def __init__(self, observation: _Optional[_Union[Observation, _Mapping]] = ..., reward: _Optional[float] = ..., terminated: _Optional[bool] = ..., truncated: _Optional[bool] = ..., auto_reset: _Optional[bool] = ..., step_count: _Optional[int] = ...) -> None: ...

class StartStreamRequest(_message.Message):
    __slots__ = ("from_frame_id",)
    FROM_FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    from_frame_id: int
    def __init__(self, from_frame_id: _Optional[int] = ...) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("env_id", "mission", "step_count", "max_steps", "frames_broadcast", "active_watchers", "last_frame_id", "agent_pos")
    ENV_ID_FIELD_NUMBER: _ClassVar[int]
    MISSION_FIELD_NUMBER: _ClassVar[int]
    STEP_COUNT_FIELD_NUMBER: _ClassVar[int]
    MAX_STEPS_FIELD_NUMBER: _ClassVar[int]
    FRAMES_BROADCAST_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_WATCHERS_FIELD_NUMBER: _ClassVar[int]
    LAST_FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_POS_FIELD_NUMBER: _ClassVar[int]
    env_id: str
    mission: str
    step_count: int
    max_steps: int
    frames_broadcast: int
    active_watchers: int
    last_frame_id: int
    agent_pos: str
    def __init__(self, env_id: _Optional[str] = ..., mission: _Optional[str] = ..., step_count: _Optional[int] = ..., max_steps: _Optional[int] = ..., frames_broadcast: _Optional[int] = ..., active_watchers: _Optional[int] = ..., last_frame_id: _Optional[int] = ..., agent_pos: _Optional[str] = ...) -> None: ...

class StatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
