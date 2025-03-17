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
            
            if controller not in ['left', 'right']:
                raise ValueError("Controller must be 'left' or 'right'")
            
            # Update the controller state with available fields
            for field in ['position', 'rotation', 'buttons', 'axes']:
                if field in data:
                    controller_states[controller][field] = data[field]
            
            # Log combined state of both controllers
            logging.warn(f"Controllers - Left: [pos:{controller_states['left']['position']}, "
                        f"\nrot:{controller_states['left']['rotation']}, "
                        f"\nbtn:{controller_states['left']['buttons']}, "
                        f"\naxes:{controller_states['left']['axes']}] | "
                        f"\n\nRight: [pos:{controller_states['right']['position']}, "
                        f"\nrot:{controller_states['right']['rotation']}, "
                        f"\nbtn:{controller_states['right']['buttons']}, "
                        f"\naxes:{controller_states['right']['axes']}]")
            
            return json.dumps({"status": "success"})
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}")
    except Exception as e:
        logging.error(f"Error processing message: {e}")

