from pykos import KOS
import asyncio
import time
from .utils.logging import setup_logger
logger = setup_logger(__name__)

async def enable_actuators(kos_instance, actuator_list):
    config_commands = []
    for actuator in actuator_list.values():
        config_commands.append(kos_instance.actuator.configure_actuator(
            actuator_id=actuator.actuator_id,
            kp=actuator.kp,
            kd=actuator.kd,
            torque_enabled=True,
        ))
    await asyncio.gather(*config_commands)


async def disable_actuators(kos_instance, actuator_list):
    disable_commands = []
    for actuator in actuator_list.values():
        disable_commands.append(
            kos_instance.actuator.configure_actuator(actuator_id=actuator.actuator_id, torque_enabled=False)
        )
    await asyncio.gather(*disable_commands)
    await asyncio.sleep(1)

