"""BOT-SORT (ByteTrack + ReID) tracker.

Adds appearance ReID features on top of ByteTrack for stronger
long-term re-identification. Higher compute but better ID stability
for ads that re-enter the frame.
"""

from __future__ import annotations

from loguru import logger
from pi_ad_blocker.config import TrackerConfig
from . import ByteTrack


class BOTSort(ByteTrack):
    """BOT-SORT: ByteTrack + appearance ReID features."""

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        logger.debug("BOT-SORT tracker initialized (ByteTrack + ReID)")