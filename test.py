import logging

# Configure the logger
logging.basicConfig(filename='process_data.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Log a message
logging.info("This is an informational message.")
logging.warning("This is a warning message.")
logging.error("This is an error message.")