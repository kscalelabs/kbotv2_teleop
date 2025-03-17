import asyncio
import websockets
import json
import logging
from .vr_cont import process_message


async def handle_connection(websocket):
    """Handle incoming WebSocket connections."""
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



