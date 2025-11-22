"""
Configuration file for Federated Learning
Edit this file to configure your FL setup
"""

# ========================
# SERVER CONFIGURATION
# ========================
SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 8080

# ========================
# CLIENT CONFIGURATION
# ========================
# Edit this on each client machine to point to the server's IP
SERVER_IP = '172.17.0.83'  # CHANGE THIS to your server machine's IP address

# ========================
# FEDERATED LEARNING PARAMETERS
# ========================
NUM_FL_ROUNDS = 10  # Number of federated learning rounds (epochs)
NUM_CLIENTS = 3  # Expected number of clients
MIN_CLIENTS = 2  # Minimum clients needed to start a round

# ========================
# MODEL TRAINING PARAMETERS
# ========================
NUM_LOCAL_EPOCHS = 5  # Local training epochs per client per FL round
BATCH_SIZE = 16  # Reduced for larger ensemble model (adjust based on GPU memory)
LEARNING_RATE = 3e-5  # Lower learning rate for fine-tuning pre-trained models
MOMENTUM = 0.9  # Not used with AdamW optimizer
NUM_CLASSES = 4

# ========================
# DATASET CONFIGURATION
# ========================
# On CLIENT machines, place your dataset in this structure:
# /home/username/federated_learning_data/
#     ├── Training/
#     │   ├── glioma_tumor/
#     │   ├── meningioma_tumor/
#     │   ├── no_tumor/
#     │   └── pituitary_tumor/
#     └── Testing/
#         ├── glioma_tumor/
#         ├── meningioma_tumor/
#         ├── no_tumor/
#         └── pituitary_tumor/

CLIENT_DATA_DIR = '/home/sudhi-sundar-dutta/Desktop/Federated/dataset'  # CHANGE THIS on each client
CLIENT_TRAIN_DIR = 'Training'
CLIENT_TEST_DIR = 'Testing'

# On SERVER machine, place validation/test dataset
SERVER_DATA_DIR = '/home/aditya/Desktop/Everything/federated_learning/dataset'
SERVER_TRAIN_DIR = 'Training'
SERVER_TEST_DIR = 'Testing'

# ========================
# MODEL SAVE PATH
# ========================
SERVER_MODEL_SAVE_PATH = '/home/aditya/Desktop/Everything/federated_learning/models'

# ========================
# NETWORK SETTINGS
# ========================
TIMEOUT = 300  # Timeout in seconds for client connections
BUFFER_SIZE = 4096  # Buffer size for socket communication
