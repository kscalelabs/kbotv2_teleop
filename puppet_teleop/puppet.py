import asyncio
import logging
import time

import colorlogging
import matplotlib.pyplot as plt
import pykos
from typing import Any

logger = logging.getLogger(__name__)

class Puppet:
    def __init__(self, kos: pykos.KOS):
        self.kos = kos
        self.actuator_ids = [11, 12, 13, 14, 35, 21, 22, 23, 24, 45]
        self.aidtoname = {
            11: "left_shoulder_pitch_03",
            12: "left_shoulder_roll_03",
            13: "left_shoulder_yaw_02",
            14: "left_elbow_02",
            35: "left_wrist_roll_02",
            21: "right_shoulder_pitch_03",
            22: "right_shoulder_roll_03",
            23: "right_shoulder_yaw_02",
            24: "right_elbow_02",
            45: "right_wrist_roll_02",
        }

    async def configure_controller(self) -> None:
        for actuator_id in self.actuator_ids:
            logger.info(
                "Configuring actuator %d name %s",
                actuator_id,
                self.aidtoname[actuator_id],
            )
            await self.kos.actuator.configure_actuator(
                actuator_id=actuator_id,
                kp=32.0,
                kd=32.0,
                zero_position=False,
                torque_enabled=True,
            )
            
async def main() -> None:
    pass
            
if __name__ == "__main__":
    asyncio.run(main())