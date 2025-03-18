import asyncio
import websockets
import json
import pybullet as p
import pybullet_data

async def send_controller_state():
    uri = "ws://localhost:8586"
    
    # Example controller state message
    message = {
        "controller": "right",
        "position": [0.1, 0.2, 0.3],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "buttons": [0, 0, 1],
        "axes": [0.5, 0.8]
    }

    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(message))
            response = await websocket.recv()
            print(f"Received response: {response}")
    except Exception as e:
        print(f"Error connecting to websocket: {e}")


if __name__ == "__main__":
    asyncio.run(send_controller_state())
