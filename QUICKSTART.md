# Federated Learning Quick Reference Guide

## 🎯 Quick Start

### On Server (Your Machine):
```bash
# 1. Find your IP address
hostname -I

# 2. Edit config.py - set SERVER_IP to your IP
nano config.py

# 3. Start server
python fl_server.py
```

### On Each Client Machine:
```bash
# 1. Copy files from server
scp aditya@server_ip:/path/model_architecture.py .
scp aditya@server_ip:/path/config.py .
scp aditya@server_ip:/path/fl_client.py .

# 2. Edit config.py
#    - Set SERVER_IP to server's IP
#    - Set CLIENT_DATA_DIR to your dataset path

# 3. Start client
python fl_client.py client_1
```

---

## 📋 File Locations

### Server Machine (Your Machine):
```
/home/aditya/Desktop/Everything/federated_learning/
├── model_architecture.py  ✓ Keep
├── config.py             ✓ Keep & Edit
├── fl_server.py          ✓ Keep
├── fl_client.py          (Optional - for testing)
├── dataset/              ✓ Your validation data
│   ├── Training/
│   └── Testing/
└── models/               (Created automatically)
```

### Client Machines:
```
/home/username/some_folder/
├── model_architecture.py  ← Copy from server
├── config.py             ← Copy & Edit
├── fl_client.py          ← Copy from server

/home/username/federated_learning_data/  ← Configure in config.py
├── Training/             ← Place dataset here
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    └── (same structure)
```

---

## ⚙️ Configuration Checklist

### In config.py - Server Machine:
- [ ] `SERVER_IP = '192.168.1.100'` ← Your server's actual IP
- [ ] `NUM_CLIENTS = 3` ← How many client machines you have
- [ ] `MIN_CLIENTS = 2` ← Minimum to start training
- [ ] `NUM_FL_ROUNDS = 10` ← Total federated rounds
- [ ] `NUM_LOCAL_EPOCHS = 5` ← Epochs per client per round

### In config.py - Client Machines:
- [ ] `SERVER_IP = '192.168.1.100'` ← Same as server's IP
- [ ] `CLIENT_DATA_DIR = '/path/to/your/data'` ← Where your dataset is

---

## 🔄 Training Flow

```
Round 1:
  Server → Sends global model → Client 1, Client 2, Client 3
  Client 1 → Trains 5 epochs on local data → Sends weights → Server
  Client 2 → Trains 5 epochs on local data → Sends weights → Server
  Client 3 → Trains 5 epochs on local data → Sends weights → Server
  Server → Averages all weights → Updates global model
  Server → Evaluates on validation set → Saves checkpoint

Round 2:
  (Repeat...)
```

---

## 🔍 Important Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_FL_ROUNDS` | 10 | Number of federated learning rounds |
| `NUM_LOCAL_EPOCHS` | 5 | Training epochs per client per round |
| `NUM_CLIENTS` | 3 | Expected number of clients |
| `MIN_CLIENTS` | 2 | Minimum clients needed to start |
| `BATCH_SIZE` | 16 | Batch size (reduced for ensemble model) |
| `LEARNING_RATE` | 3e-5 | Learning rate (lower for fine-tuning) |
| `SERVER_PORT` | 8080 | Port for communication |
| `TIMEOUT` | 300 | Connection timeout (seconds) |

**Note**: Model is an Ensemble (Swin + DeiT + ConvNeXt) with ~187M parameters and ~750 MB size.

---

## 🐛 Troubleshooting

### Server won't start:
```bash
# Check if port is already in use
netstat -tulpn | grep 8080

# Kill process using port
sudo kill -9 <PID>
```

### Client can't connect:
```bash
# Test connectivity
ping server_ip
telnet server_ip 8080

# Check firewall
sudo ufw status
sudo ufw allow 8080/tcp
```

### Dataset not found:
- Check path in `config.py` → `CLIENT_DATA_DIR`
- Verify folder structure matches expected format
- Ensure folders are named exactly: `Training`, `Testing`, `glioma_tumor`, etc.

### GPU out of memory:
- Reduce `BATCH_SIZE` in `config.py` (try 8 or 4)
- Ensemble model requires 8-12 GB GPU memory
- Use CPU if needed (slower): Device will auto-select

---

## 📊 Expected Output

### Server Console:
```
============================================================
Starting Federated Learning Server
============================================================
Server listening on 0.0.0.0:8080
Expected clients: 3

============================================================
FEDERATED LEARNING ROUND 1/10
============================================================
Waiting for clients to connect...
Client 1 connected from ('192.168.1.101', 54321)
  → Sent global model to client
  ← Received trained weights from client (trained on 2500 samples)
...
Aggregating weights using FedAvg...
[Round 1] Global Model - Val Acc: 0.8234, Val Loss: 0.4523
```

### Client Console:
```
============================================================
Starting Federated Learning Client client_1
============================================================
Server: 192.168.1.100:8080
Local dataset size: 2500 samples

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

---

## 💡 Tips

1. **Always start server first**, then clients
2. **All machines must be on same network**
3. **Test with 1 client first** before adding more
4. **Keep datasets similar size** across clients for best results
5. **Monitor server logs** to track training progress
6. **Save checkpoints** are in `models/` directory
7. **Final model** is `models/final_global_model.pth`

---

## 🚀 Quick Commands

```bash
# Install dependencies (IMPORTANT: includes timm for ensemble model)
pip install -r requirements.txt
# or: pip install torch torchvision timm numpy scikit-learn matplotlib Pillow

# Find your IP
hostname -I

# Start server
python fl_server.py

# Start client
python fl_client.py client_1

# Copy files to client (MUST include updated model_architecture.py)
scp model_architecture.py config.py fl_client.py requirements.txt user@client_ip:/path/

# Check if server is running
netstat -tulpn | grep 8080

# Test connectivity from client
ping server_ip
```

**⚠️ CRITICAL**: The `timm` library is required for the ensemble model. First run will download ~350-400 MB of pre-trained weights.

---

## 📁 What to Copy to Client Machines

**Minimum required files:**
1. `model_architecture.py` ⚠️ **CRITICAL - Updated for Ensemble Model**
2. `config.py` (edited with correct SERVER_IP and CLIENT_DATA_DIR)
3. `fl_client.py`
4. `requirements.txt` (for installing dependencies including timm)

**Do NOT copy:**
- `dataset/` folder (clients should have their own data)
- `fl_server.py` (only needed on server)
- `models/` folder (only on server)

**Important**: The `model_architecture.py` file has been completely rewritten to use an Ensemble Model instead of MobileNetV2. You MUST copy the updated version to all clients.

---

## ✅ Pre-Flight Checklist

Before starting training:

**Server:**
- [ ] config.py has correct SERVER_IP
- [ ] Dataset is in dataset/Training and dataset/Testing
- [ ] Port 8080 is open (firewall)
- [ ] Server script is ready: `python fl_server.py`

**Each Client:**
- [ ] Copied model_architecture.py (UPDATED version), config.py, fl_client.py
- [ ] Installed timm library: `pip install timm`
- [ ] config.py has correct SERVER_IP and CLIENT_DATA_DIR
- [ ] Dataset is placed in CLIENT_DATA_DIR
- [ ] Can ping server: `ping server_ip`
- [ ] Has sufficient GPU memory (8-12 GB) or reduced BATCH_SIZE
- [ ] Ready to run: `python fl_client.py client_X`

**Network:**
- [ ] All machines on same network
- [ ] Firewall allows connections on port 8080
- [ ] Tested connectivity between machines

---

## 🎓 Understanding the Architecture

**Traditional ML (Centralized):**
```
All data → Single machine → Train model → Deploy
```

**Federated Learning:**
```
Data stays on clients → Models train locally → 
Weights aggregated on server → Global model improves
```

**Key Differences:**
- **Privacy**: Data never leaves client machines
- **Scalability**: Can add more clients
- **Communication**: Network overhead between rounds
- **Training Time**: Slower due to network + multiple machines

---

**For detailed information, see README.md**
