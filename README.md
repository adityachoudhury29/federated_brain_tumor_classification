# Federated Learning for Brain Tumor Classification

A federated learning implementation of MobileNetV2 for brain tumor classification using distributed datasets across multiple client machines.

## 📋 Overview

This system implements **Federated Learning (FL)** where:
- **Server** hosts the global model and coordinates training
- **Clients** train the model on their local datasets without sharing raw data
- Model weights are aggregated using **FedAvg (Federated Averaging)**
- Privacy is preserved as data never leaves client machines

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         FL SERVER                            │
│  - Manages global MobileNetV2 model                         │
│  - Coordinates federated rounds                             │
│  - Aggregates client weights (FedAvg)                       │
│  - Evaluates on validation set                              │
└─────────────────────────────────────────────────────────────┘
                    ▲                    ▲
                    │ Weights            │ Weights
                    │ Exchange           │ Exchange
                    ▼                    ▼
        ┌──────────────────┐  ┌──────────────────┐
        │    CLIENT 1       │  │    CLIENT 2       │
        │  - Local dataset  │  │  - Local dataset  │
        │  - Local training │  │  - Local training │
        └──────────────────┘  └──────────────────┘
```

## 📁 File Structure

```
federated_learning/
├── model_architecture.py   # MobileNetV2 model definition
├── config.py              # Configuration parameters
├── fl_server.py          # Server-side code
├── fl_client.py          # Client-side code
├── mn-new.py             # Original centralized training (for reference)
├── dataset/              # Server's validation dataset
│   ├── Training/
│   └── Testing/
└── models/               # Saved model checkpoints (created by server)
```

## 🔧 Setup Instructions

### Prerequisites

Install required packages on **all machines** (server + clients):

```bash
pip install torch torchvision numpy scikit-learn matplotlib
```

### Step 1: Configure Network Settings

1. **Find your server's IP address:**
   ```bash
   # On server machine
   hostname -I
   # or
   ip addr show
   ```

2. **Edit `config.py` on ALL machines:**
   ```python
   # On SERVER machine:
   SERVER_IP = '192.168.1.100'  # Your server's actual IP
   
   # On CLIENT machines:
   SERVER_IP = '192.168.1.100'  # Same as server's IP
   ```

### Step 2: Set Up Datasets

#### On Server Machine:
Keep your validation dataset in:
```
/home/aditya/Desktop/Everything/federated_learning/dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

#### On Each Client Machine:

1. **Create the dataset directory:**
   ```bash
   mkdir -p ~/federated_learning_data
   ```

2. **Copy dataset to client** (or create it there):
   ```
   /home/username/federated_learning_data/
   ├── Training/
   │   ├── glioma_tumor/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   ├── meningioma_tumor/
   │   ├── no_tumor/
   │   └── pituitary_tumor/
   └── Testing/
       ├── glioma_tumor/
       ├── meningioma_tumor/
       ├── no_tumor/
       └── pituitary_tumor/
   ```

3. **Update `config.py` on each client:**
   ```python
   CLIENT_DATA_DIR = '/home/username/federated_learning_data'  # Change 'username' to actual username
   ```

### Step 3: Copy Required Files to Client Machines

Copy these files to **each client machine**:
- `model_architecture.py`
- `config.py` (with correct paths)
- `fl_client.py`

You can use `scp`:
```bash
# From server, copy to client
scp model_architecture.py config.py fl_client.py username@client_ip:/path/to/client/folder/
```

## 🚀 Running the Federated Learning System

### Important: Start Order

**ALWAYS start the server first, then clients!**

### Step 1: Start the Server

On your server machine:
```bash
cd /home/aditya/Desktop/Everything/federated_learning
python fl_server.py
```

The server will:
- Initialize the global model
- Wait for clients to connect
- Display "Waiting for clients to connect..."

### Step 2: Start Clients

On each client machine:
```bash
python fl_client.py client_1
```

Or without command line argument:
```bash
python fl_client.py
# Then enter client ID when prompted: client_1
```

**Start each client with a unique ID:**
- Machine 1: `python fl_client.py client_1`
- Machine 2: `python fl_client.py client_2`
- Machine 3: `python fl_client.py client_3`

### Step 3: Training Process

Once minimum clients connect, the training begins:

**Each FL Round:**
1. Server sends global model to all clients
2. Each client trains on local data (5 epochs by default)
3. Clients send trained weights back to server
4. Server aggregates weights using FedAvg
5. Server evaluates global model on validation set
6. Repeat for next round

## ⚙️ Configuration Parameters

Edit `config.py` to customize:

```python
# Federated Learning
NUM_FL_ROUNDS = 10        # Number of federated rounds
NUM_CLIENTS = 3           # Expected number of clients
MIN_CLIENTS = 2           # Minimum clients to start

# Local Training
NUM_LOCAL_EPOCHS = 5      # Training epochs per client per round
BATCH_SIZE = 32
LEARNING_RATE = 0.01
```

## 📊 Output and Results

### Server Output:
```
============================================================
FEDERATED LEARNING ROUND 1/10
============================================================
Waiting for clients to connect...
Client 1 connected from ('192.168.1.101', 54321)
  → Sent global model to client
  ← Received trained weights from client
...

Aggregating weights using FedAvg...
Global model updated with aggregated weights

[Round 1] Global Model - Val Acc: 0.8523, Val Loss: 0.4127
```

### Client Output:
```
============================================================
Client client_1 - FL Round 1
============================================================
Connected to server at 192.168.1.100:8080
✓ Global model loaded successfully

Training on local data for 5 epochs...
  Epoch 1/5 - Loss: 0.6234, Acc: 0.7156
  Epoch 2/5 - Loss: 0.4567, Acc: 0.8234
  ...
✓ Local training completed
✓ Weights sent successfully
```

### Saved Models:
Models are saved in `/home/aditya/Desktop/Everything/federated_learning/models/`:
- `global_model_round_1.pth`
- `global_model_round_2.pth`
- ...
- `final_global_model.pth`

## 🔍 Monitoring and Debugging

### Check if Server is Listening:
```bash
# On server machine
netstat -tulpn | grep 8080
```

### Test Network Connectivity:
```bash
# From client machine
ping server_ip
telnet server_ip 8080
```

### Common Issues:

**1. "Connection refused"**
   - Make sure server is running first
   - Check firewall settings
   - Verify IP address in config.py

**2. "Dataset not found"**
   - Verify dataset path in config.py
   - Check folder structure matches expected format

**3. "Timeout waiting for clients"**
   - Increase TIMEOUT in config.py
   - Start clients faster after server starts
   - Check network connectivity

**4. Firewall blocking connections:**
   ```bash
   # On server machine, allow incoming connections
   sudo ufw allow 8080/tcp
   # or disable firewall temporarily for testing
   sudo ufw disable
   ```

## 📈 Federated Learning vs Centralized Training

| Aspect | Centralized (mn-new.py) | Federated (this system) |
|--------|------------------------|-------------------------|
| Data Location | All data on one machine | Distributed across clients |
| Privacy | Data must be shared | Data stays on client machines |
| Epochs | ~50 model epochs | 10 FL rounds × 5 local epochs |
| Training Time | Faster (single machine) | Slower (network communication) |
| Scalability | Limited by single machine | Scales with more clients |

## 🔒 Privacy Benefits

- **Data Privacy**: Raw data never leaves client machines
- **Model Privacy**: Only model weights are shared
- **Secure Aggregation**: Server only sees aggregated weights
- **Compliance**: Suitable for sensitive medical data

## 🎯 Next Steps

1. **Test with more clients**: Add more machines to the network
2. **Experiment with parameters**: Try different FL rounds, local epochs
3. **Add differential privacy**: Implement noise addition for stronger privacy
4. **Implement client selection**: Randomly select subset of clients per round
5. **Add evaluation metrics**: Implement confusion matrix on server

## 📝 Notes

- **FL Round ≠ Model Epoch**: One FL round contains multiple local training epochs
- **Network Speed**: Training speed depends on network bandwidth
- **Synchronous FL**: All clients must participate in each round
- **IID Assumption**: This implementation assumes similar data distribution across clients

## 🆘 Support

If you encounter issues:
1. Check all machines are on same network
2. Verify IP addresses are correct
3. Ensure datasets are properly placed
4. Check firewall settings
5. Review logs for error messages

---

**Author**: Federated Learning Implementation for Brain Tumor Classification  
**Date**: October 26, 2025
