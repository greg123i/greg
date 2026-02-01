
import logging
import os

# Configure logging to file
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'greg_debug.log')
logging.basicConfig(filename=log_file, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def log_debug(msg):
    logging.debug(msg)
    # Also print to console for terminal visibility
    print(f"[DEBUG] {msg}")

def log_error(msg):
    logging.error(msg)
    print(f"[ERROR] {msg}")
