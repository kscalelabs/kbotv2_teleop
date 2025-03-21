import asyncio
import websockets
import json
import logging

logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Store the latest state of both controllers
controller_states = {
    'left': {'position': None, 'rotation': None, 'buttons': None, 'axes': None},
    'right': {'position': None, 'rotation': None, 'buttons': None, 'axes': None}
}

async def process_message(message):
    """
    Process the received message and update controller states.
    """
    try:
        data = json.loads(message)
        
        if 'controller' in data:
            controller = data['controller']

            breakpoint()
            
            if controller not in ['left', 'right']:
                raise ValueError("Controller must be 'left' or 'right'")
            
            # Update the controller state with available fields
            for field in ['position', 'rotation', 'buttons', 'axes']:
                if field in data:
                    controller_states[controller][field] = data[field]

            logging.warn("starting")

            logging.warn(controller_states['left']['position'])
            logging.warn(controller_states['right']['position'])
            # print(controller_states['left']['rotation'])
            # print(controller_states['right']['rotation'])
            if controller_states['left']['buttons']:
                logging.warn(controller_states['left']['buttons'][0]["pressed"])
            if controller_states['right']['buttons']:
                logging.warn(controller_states['right']['buttons'][0]["pressed"])
            # Log combined state of both controllers
            # logging.warn(f"Controllers - Left: [pos:{controller_states['left']['position']}, "
            #             f"\nrot:{controller_states['left']['rotation']}, "
            #             f"\nbtn:{controller_states['left']['buttons']}, "
            #             f"\naxes:{controller_states['left']['axes']}] | "
            #             f"\n\nRight: [pos:{controller_states['right']['position']}, "
            #             f"\nrot:{controller_states['right']['rotation']}, "
            #             f"\nbtn:{controller_states['right']['buttons']}, "
            #             f"\naxes:{controller_states['right']['axes']}]")
            
            return json.dumps({"status": "success"})
        

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}")
    except Exception as e:
        logging.error(f"Error processing message: {e}")

async def handle_connection(websocket):
    """Handle incoming WebSocket connections."""
    print("trying)")
    try:
        async for message in websocket:
            logging.info("Received connection")
            
            # Process the message
            response = await process_message(message)
            
            # Send response back
            await websocket.send(response)
            logging.info(f"Sent response: {response}")
            
    except websockets.exceptions.ConnectionClosed:
        logging.info("Client disconnected")
    except Exception as e:
        logging.error(f"Error handling connection: {e}")

async def main():
    """Start the WebSocket server."""
    server = await websockets.serve(
        handle_connection,
        "localhost",
        8586
    )
    logging.info("WebSocket server started on ws://localhost:8586")
    
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())