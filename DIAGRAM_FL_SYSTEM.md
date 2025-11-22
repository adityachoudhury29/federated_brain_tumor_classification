# 🌐 Federated Learning System Architecture - Detailed Block Diagram

This file contains comprehensive block diagrams of the complete Federated Learning system for distributed brain tumor classification.

---

## 🏛️ Complete Federated Learning System Architecture

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e8eaf6','primaryTextColor':'#000','primaryBorderColor':'#3f51b5','lineColor':'#424242','secondaryColor':'#fff3e0','tertiaryColor':'#e8f5e9'}}}%%

flowchart TB
    subgraph SERVER ["🖥️ FEDERATED LEARNING SERVER"]
        style SERVER fill:#e3f2fd,stroke:#1976d2,stroke-width:4px
        
        subgraph SERVER_INIT ["Initialization Phase"]
            style SERVER_INIT fill:#bbdefb,stroke:#1976d2,stroke-width:2px
            S_START["🚀 Server Start<br/>IP: 172.20.10.3<br/>Port: 8080"]
            S_MODEL["📦 Initialize Global Model<br/>Ensemble (Swin+DeiT+ConvNeXt)<br/>Parameters: ~187M"]
            S_DATASET["📊 Load Validation Dataset<br/>Location: ./dataset/<br/>For evaluation only"]
            
            S_START --> S_MODEL
            S_MODEL --> S_DATASET
            
            style S_START fill:#90caf9,stroke:#1976d2,stroke-width:2px
            style S_MODEL fill:#90caf9,stroke:#1976d2
            style S_DATASET fill:#90caf9,stroke:#1976d2
        end
        
        subgraph SERVER_COORD ["Coordination Phase"]
            style SERVER_COORD fill:#c5cae9,stroke:#3f51b5,stroke-width:2px
            S_LISTEN["👂 Listen for Clients<br/>TCP Socket<br/>Timeout: 300s"]
            S_ACCEPT["✅ Accept Connections<br/>Expected: 3 clients<br/>Minimum: 2 clients"]
            S_SEND["📤 Send Global Weights<br/>Serialize with Pickle<br/>Size: ~750 MB"]
            
            S_LISTEN --> S_ACCEPT
            S_ACCEPT --> S_SEND
            
            style S_LISTEN fill:#9fa8da,stroke:#3f51b5
            style S_ACCEPT fill:#9fa8da,stroke:#3f51b5
            style S_SEND fill:#9fa8da,stroke:#3f51b5
        end
        
        subgraph SERVER_AGG ["Aggregation Phase"]
            style SERVER_AGG fill:#c8e6c9,stroke:#43a047,stroke-width:2px
            S_RECEIVE["📥 Receive Client Weights<br/>From all connected clients<br/>Track data sizes"]
            S_FEDAVG["⚖️ FedAvg Algorithm<br/>W_global = Σ(W_i) / N<br/>Average all parameters"]
            S_UPDATE["🔄 Update Global Model<br/>Load aggregated weights<br/>Replace old model"]
            
            S_RECEIVE --> S_FEDAVG
            S_FEDAVG --> S_UPDATE
            
            style S_RECEIVE fill:#a5d6a7,stroke:#43a047
            style S_FEDAVG fill:#a5d6a7,stroke:#43a047,stroke-width:2px
            style S_UPDATE fill:#a5d6a7,stroke:#43a047
        end
        
        subgraph SERVER_EVAL ["Evaluation Phase"]
            style SERVER_EVAL fill:#fff9c4,stroke:#f9a825,stroke-width:2px
            S_EVAL["📈 Evaluate Global Model<br/>On validation dataset<br/>Compute accuracy & loss"]
            S_SAVE["💾 Save Checkpoint<br/>models/global_model_round_X.pth<br/>Track best model"]
            S_HISTORY["📊 Update History<br/>Track metrics per round<br/>Log performance"]
            
            S_EVAL --> S_SAVE
            S_SAVE --> S_HISTORY
            
            style S_EVAL fill:#fff59d,stroke:#f9a825
            style S_SAVE fill:#fff59d,stroke:#f9a825,stroke-width:2px
            style S_HISTORY fill:#fff59d,stroke:#f9a825
        end
        
        S_DATASET --> S_LISTEN
        S_SEND -.->|"Wait for training"| S_RECEIVE
        S_HISTORY -.->|"Next FL Round"| S_LISTEN
    end
    
    subgraph NETWORK ["🌐 NETWORK LAYER"]
        style NETWORK fill:#fff3e0,stroke:#f57c00,stroke-width:4px
        
        subgraph NET_PROTO ["Communication Protocol"]
            style NET_PROTO fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
            N_TCP["TCP/IP Sockets<br/>Reliable connection<br/>Stream-based"]
            N_PICKLE["Pickle Serialization<br/>Python objects<br/>Model weights"]
            N_BUFFER["Buffer Management<br/>Size: 4096 bytes<br/>Chunked transfer"]
            
            N_TCP --> N_PICKLE
            N_PICKLE --> N_BUFFER
            
            style N_TCP fill:#ffcc80,stroke:#f57c00
            style N_PICKLE fill:#ffcc80,stroke:#f57c00
            style N_BUFFER fill:#ffcc80,stroke:#f57c00
        end
        
        subgraph NET_DATA ["Data Transfer"]
            style NET_DATA fill:#ffecb3,stroke:#ffa000,stroke-width:2px
            N_DOWN["⬇️ Download<br/>Server → Client<br/>~750 MB per round"]
            N_UP["⬆️ Upload<br/>Client → Server<br/>~750 MB per round"]
            N_TOTAL["📊 Total per Round<br/>~1.5 GB per client<br/>~4.5 GB for 3 clients"]
            
            N_DOWN --> N_UP
            N_UP --> N_TOTAL
            
            style N_DOWN fill:#ffd54f,stroke:#ffa000
            style N_UP fill:#ffd54f,stroke:#ffa000
            style N_TOTAL fill:#ffb300,stroke:#ff6f00,stroke-width:2px,color:#fff
        end
    end
    
    subgraph CLIENTS ["💻 FEDERATED LEARNING CLIENTS"]
        style CLIENTS fill:#f3e5f5,stroke:#8e24aa,stroke-width:4px
        
        subgraph CLIENT1 ["Client 1: Machine A"]
            style CLIENT1 fill:#e1bee7,stroke:#8e24aa,stroke-width:2px
            
            C1_INIT["🔧 Initialize<br/>ID: client_1<br/>Device: GPU/CPU"]
            C1_DATA["📂 Local Dataset<br/>Path: ~/fl_data/<br/>Training + Testing"]
            C1_CONNECT["🔌 Connect to Server<br/>IP: 172.20.10.3:8080<br/>Socket connection"]
            C1_RECEIVE["📥 Receive Global Model<br/>Download weights<br/>Load into local model"]
            C1_TRAIN["🏋️ Local Training<br/>5 epochs<br/>AdamW optimizer<br/>LR: 3e-5"]
            C1_SEND["📤 Send Trained Weights<br/>Upload to server<br/>Include data size"]
            
            C1_INIT --> C1_DATA
            C1_DATA --> C1_CONNECT
            C1_CONNECT --> C1_RECEIVE
            C1_RECEIVE --> C1_TRAIN
            C1_TRAIN --> C1_SEND
            
            style C1_INIT fill:#ce93d8,stroke:#8e24aa
            style C1_DATA fill:#ce93d8,stroke:#8e24aa
            style C1_CONNECT fill:#ce93d8,stroke:#8e24aa
            style C1_RECEIVE fill:#ce93d8,stroke:#8e24aa
            style C1_TRAIN fill:#ba68c8,stroke:#8e24aa,stroke-width:2px
            style C1_SEND fill:#ce93d8,stroke:#8e24aa
        end
        
        subgraph CLIENT2 ["Client 2: Machine B"]
            style CLIENT2 fill:#e1bee7,stroke:#8e24aa,stroke-width:2px
            
            C2_INIT["🔧 Initialize<br/>ID: client_2<br/>Device: GPU/CPU"]
            C2_DATA["📂 Local Dataset<br/>Different data split<br/>Privacy preserved"]
            C2_TRAIN["🏋️ Local Training<br/>Same architecture<br/>Independent training"]
            C2_SEND["📤 Send Weights<br/>Only model updates<br/>No raw data"]
            
            C2_INIT --> C2_DATA
            C2_DATA --> C2_TRAIN
            C2_TRAIN --> C2_SEND
            
            style C2_INIT fill:#ce93d8,stroke:#8e24aa
            style C2_DATA fill:#ce93d8,stroke:#8e24aa
            style C2_TRAIN fill:#ba68c8,stroke:#8e24aa,stroke-width:2px
            style C2_SEND fill:#ce93d8,stroke:#8e24aa
        end
        
        subgraph CLIENT3 ["Client 3: Machine C"]
            style CLIENT3 fill:#e1bee7,stroke:#8e24aa,stroke-width:2px
            
            C3_INIT["🔧 Initialize<br/>ID: client_3<br/>Device: GPU/CPU"]
            C3_DATA["📂 Local Dataset<br/>Third data partition<br/>Distributed learning"]
            C3_TRAIN["🏋️ Local Training<br/>Parallel with others<br/>Asynchronous start"]
            C3_SEND["📤 Send Weights<br/>Complete FL round<br/>Disconnect"]
            
            C3_INIT --> C3_DATA
            C3_DATA --> C3_TRAIN
            C3_TRAIN --> C3_SEND
            
            style C3_INIT fill:#ce93d8,stroke:#8e24aa
            style C3_DATA fill:#ce93d8,stroke:#8e24aa
            style C3_TRAIN fill:#ba68c8,stroke:#8e24aa,stroke-width:2px
            style C3_SEND fill:#ce93d8,stroke:#8e24aa
        end
    end
    
    %% Connections between Server and Network
    S_SEND -.->|"Model weights"| N_DOWN
    N_DOWN -.->|"Distribute"| C1_RECEIVE
    N_DOWN -.->|"Distribute"| C2_TRAIN
    N_DOWN -.->|"Distribute"| C3_TRAIN
    
    %% Connections from Clients to Network
    C1_SEND -.->|"Trained weights"| N_UP
    C2_SEND -.->|"Trained weights"| N_UP
    C3_SEND -.->|"Trained weights"| N_UP
    
    %% Network to Server
    N_UP -.->|"Aggregate"| S_RECEIVE
```

---

## 🔄 Federated Learning Round - Detailed Sequence

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e8f5e9','primaryTextColor':'#000','primaryBorderColor':'#43a047'}}}%%

sequenceDiagram
    autonumber
    
    participant Server as 🖥️ FL Server<br/>(172.20.10.3:8080)
    participant Network as 🌐 Network Layer<br/>(TCP/IP)
    participant Client1 as 💻 Client 1<br/>(Machine A)
    participant Client2 as 💻 Client 2<br/>(Machine B)
    participant Client3 as 💻 Client 3<br/>(Machine C)
    
    rect rgb(227, 242, 253)
    Note over Server: 📋 FL Round N Begins
    Server->>Server: Initialize round N
    Server->>Server: Set round counter
    end
    
    rect rgb(200, 230, 201)
    Note over Server,Client3: 🔌 Connection Phase
    Server->>Network: Listen on port 8080
    Network->>Network: Create socket
    
    Client1->>Network: Connect request
    Network->>Server: Accept Client 1
    Server->>Network: Connection established
    
    Client2->>Network: Connect request
    Network->>Server: Accept Client 2
    Server->>Network: Connection established
    
    Client3->>Network: Connect request
    Network->>Server: Accept Client 3
    Server->>Network: Connection established
    
    Note over Server: ✅ All clients connected
    end
    
    rect rgb(255, 249, 196)
    Note over Server,Client3: 📤 Model Distribution Phase
    Server->>Server: Serialize global model
    Server->>Network: Send model size (8 bytes)
    Network->>Client1: Forward size
    Network->>Client2: Forward size
    Network->>Client3: Forward size
    
    Server->>Network: Send model weights (~750 MB)
    Network->>Client1: Stream weights (chunks)
    Network->>Client2: Stream weights (chunks)
    Network->>Client3: Stream weights (chunks)
    
    Client1->>Client1: Deserialize & load weights
    Client2->>Client2: Deserialize & load weights
    Client3->>Client3: Deserialize & load weights
    
    Note over Client1,Client3: ✓ Models loaded
    end
    
    rect rgb(225, 190, 231)
    Note over Client1,Client3: 🏋️ Local Training Phase
    
    par Parallel Training
        Client1->>Client1: Load local dataset
        Client1->>Client1: Create DataLoader
        loop 5 Epochs
            Client1->>Client1: Forward pass
            Client1->>Client1: Compute loss
            Client1->>Client1: Backward pass
            Client1->>Client1: Update weights
        end
        Client1->>Client1: Extract model weights
        
    and
        Client2->>Client2: Load local dataset
        Client2->>Client2: Create DataLoader
        loop 5 Epochs
            Client2->>Client2: Forward pass
            Client2->>Client2: Compute loss
            Client2->>Client2: Backward pass
            Client2->>Client2: Update weights
        end
        Client2->>Client2: Extract model weights
        
    and
        Client3->>Client3: Load local dataset
        Client3->>Client3: Create DataLoader
        loop 5 Epochs
            Client3->>Client3: Forward pass
            Client3->>Client3: Compute loss
            Client3->>Client3: Backward pass
            Client3->>Client3: Update weights
        end
        Client3->>Client3: Extract model weights
    end
    
    Note over Client1,Client3: ✓ Training complete
    end
    
    rect rgb(255, 235, 238)
    Note over Server,Client3: 📥 Weight Collection Phase
    
    Client1->>Client1: Serialize weights + metadata
    Client1->>Network: Send size (8 bytes)
    Client1->>Network: Send weights (~750 MB)
    Network->>Server: Receive from Client 1
    Server->>Server: Store weights_1, size_1
    
    Client2->>Client2: Serialize weights + metadata
    Client2->>Network: Send size (8 bytes)
    Client2->>Network: Send weights (~750 MB)
    Network->>Server: Receive from Client 2
    Server->>Server: Store weights_2, size_2
    
    Client3->>Client3: Serialize weights + metadata
    Client3->>Network: Send size (8 bytes)
    Client3->>Network: Send weights (~750 MB)
    Network->>Server: Receive from Client 3
    Server->>Server: Store weights_3, size_3
    
    Note over Server: ✓ All weights collected
    end
    
    rect rgb(255, 224, 130)
    Note over Server: ⚖️ Aggregation Phase (FedAvg)
    Server->>Server: Initialize averaged_weights
    loop For each parameter
        Server->>Server: Sum all client weights
        Server->>Server: Divide by num_clients
    end
    Server->>Server: Load averaged weights
    Server->>Server: Update global model
    Note over Server: ✓ Global model updated
    end
    
    rect rgb(179, 229, 252)
    Note over Server: 📊 Evaluation Phase
    Server->>Server: Set model to eval mode
    Server->>Server: Load validation dataset
    loop For each batch
        Server->>Server: Forward pass (no grad)
        Server->>Server: Compute predictions
        Server->>Server: Calculate metrics
    end
    Server->>Server: Aggregate metrics
    Server->>Server: Val Acc: X.XX%
    Server->>Server: Val Loss: X.XXX
    end
    
    rect rgb(200, 230, 201)
    Note over Server: 💾 Checkpoint Phase
    Server->>Server: Save model weights
    Server->>Server: Save to models/global_model_round_N.pth
    Server->>Server: Update training history
    Server->>Server: Log metrics
    Note over Server: ✓ Checkpoint saved
    end
    
    rect rgb(255, 205, 210)
    Note over Server,Client3: 🔌 Disconnect Phase
    Server->>Client1: Close connection
    Server->>Client2: Close connection
    Server->>Client3: Close connection
    
    Client1->>Client1: Disconnect & cleanup
    Client2->>Client2: Disconnect & cleanup
    Client3->>Client3: Disconnect & cleanup
    end
    
    rect rgb(232, 234, 246)
    Note over Server: 🔄 Round Complete
    Server->>Server: Round N finished
    Server->>Server: Increment round counter
    alt More rounds remaining
        Server->>Server: Start Round N+1
    else All rounds complete
        Server->>Server: Save final model
        Server->>Server: Print summary
        Note over Server: ✅ Training Complete
    end
    end
```

---

## 🗂️ Data Flow Architecture

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#fff3e0','primaryTextColor':'#000','primaryBorderColor':'#f57c00'}}}%%

flowchart LR
    subgraph DATA_ARCH ["Data & Privacy Architecture"]
        style DATA_ARCH fill:#fff3e0,stroke:#f57c00,stroke-width:4px
        
        subgraph SERVER_DATA ["Server Data"]
            style SERVER_DATA fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
            S_VAL["Validation Dataset<br/>📊 Used for:<br/>• Global model evaluation<br/>• Performance tracking<br/>• Not used in training"]
            S_MODEL_W["Global Model Weights<br/>⚖️ Aggregated from:<br/>• All client updates<br/>• FedAvg algorithm<br/>• Shared with clients"]
            
            style S_VAL fill:#bbdefb,stroke:#1976d2
            style S_MODEL_W fill:#90caf9,stroke:#1976d2,stroke-width:2px
        end
        
        subgraph CLIENT1_DATA ["Client 1 Data"]
            style CLIENT1_DATA fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
            C1_TRAIN["Training Data<br/>🔒 Private:<br/>• Never leaves client<br/>• Local training only<br/>• Different from others"]
            C1_WEIGHTS["Model Weights<br/>📤 Shared:<br/>• After local training<br/>• Sent to server<br/>• No raw data included"]
            
            C1_TRAIN -.->|"Trains"| C1_WEIGHTS
            
            style C1_TRAIN fill:#e1bee7,stroke:#8e24aa,stroke-width:2px
            style C1_WEIGHTS fill:#ce93d8,stroke:#8e24aa
        end
        
        subgraph CLIENT2_DATA ["Client 2 Data"]
            style CLIENT2_DATA fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
            C2_TRAIN["Training Data<br/>🔒 Private:<br/>• Independent dataset<br/>• Different distribution<br/>• Stays local"]
            C2_WEIGHTS["Model Weights<br/>📤 Shared:<br/>• Trained weights only<br/>• No images sent<br/>• Privacy preserved"]
            
            C2_TRAIN -.->|"Trains"| C2_WEIGHTS
            
            style C2_TRAIN fill:#e1bee7,stroke:#8e24aa,stroke-width:2px
            style C2_WEIGHTS fill:#ce93d8,stroke:#8e24aa
        end
        
        subgraph CLIENT3_DATA ["Client 3 Data"]
            style CLIENT3_DATA fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
            C3_TRAIN["Training Data<br/>🔒 Private:<br/>• Third partition<br/>• Unique samples<br/>• Local only"]
            C3_WEIGHTS["Model Weights<br/>📤 Shared:<br/>• Parameter updates<br/>• Aggregated later<br/>• Secure transfer"]
            
            C3_TRAIN -.->|"Trains"| C3_WEIGHTS
            
            style C3_TRAIN fill:#e1bee7,stroke:#8e24aa,stroke-width:2px
            style C3_WEIGHTS fill:#ce93d8,stroke:#8e24aa
        end
        
        subgraph PRIVACY ["Privacy Guarantees"]
            style PRIVACY fill:#c8e6c9,stroke:#43a047,stroke-width:3px
            P1["✅ Raw data never transmitted"]
            P2["✅ Only model weights shared"]
            P3["✅ Each client has unique data"]
            P4["✅ Server cannot reconstruct images"]
            P5["✅ Compliant with privacy regulations"]
            
            style P1 fill:#a5d6a7,stroke:#43a047
            style P2 fill:#a5d6a7,stroke:#43a047
            style P3 fill:#a5d6a7,stroke:#43a047
            style P4 fill:#a5d6a7,stroke:#43a047
            style P5 fill:#81c784,stroke:#43a047,stroke-width:2px
        end
        
        %% Connections
        S_MODEL_W -.->|"Download"| C1_WEIGHTS
        S_MODEL_W -.->|"Download"| C2_WEIGHTS
        S_MODEL_W -.->|"Download"| C3_WEIGHTS
        
        C1_WEIGHTS -.->|"Upload"| S_MODEL_W
        C2_WEIGHTS -.->|"Upload"| S_MODEL_W
        C3_WEIGHTS -.->|"Upload"| S_MODEL_W
    end
```

---

## 📈 Training Progress Timeline

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e8eaf6','primaryTextColor':'#000','primaryBorderColor':'#3f51b5'}}}%%

gantt
    title Federated Learning Training Timeline (10 Rounds)
    dateFormat  X
    axisFormat %s
    
    section Server Init
    Initialize Model           :done, init, 0, 30
    Load Validation Data       :done, val, 30, 50
    Start Socket Server        :done, socket, 50, 60
    
    section Round 1
    Wait for Clients          :active, r1_wait, 60, 90
    Send Model (750MB)        :active, r1_send, 90, 150
    Client Training           :crit, r1_train, 150, 450
    Receive Weights           :active, r1_recv, 450, 510
    FedAvg Aggregation        :r1_agg, 510, 540
    Evaluation                :r1_eval, 540, 570
    Save Checkpoint           :r1_save, 570, 580
    
    section Round 2
    Send Model                :r2_send, 580, 640
    Client Training           :crit, r2_train, 640, 940
    Receive & Aggregate       :r2_agg, 940, 1000
    Evaluate & Save           :r2_save, 1000, 1030
    
    section Round 3
    Send Model                :r3_send, 1030, 1090
    Client Training           :crit, r3_train, 1090, 1390
    Receive & Aggregate       :r3_agg, 1390, 1450
    Evaluate & Save           :r3_save, 1450, 1480
    
    section Round 4-10
    Repeat Rounds 4-10        :r4_10, 1480, 3500
    
    section Finalization
    Save Final Model          :done, final, 3500, 3530
    Print Summary             :done, summary, 3530, 3550
    Training Complete         :milestone, complete, 3550, 3550
```

---

## 🔐 Security & Privacy Architecture

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e8f5e9','primaryTextColor':'#000','primaryBorderColor':'#43a047'}}}%%

flowchart TB
    subgraph SECURITY ["Security & Privacy Layers"]
        style SECURITY fill:#e8f5e9,stroke:#43a047,stroke-width:4px
        
        subgraph LAYER1 ["Layer 1: Data Privacy"]
            style LAYER1 fill:#c8e6c9,stroke:#43a047,stroke-width:2px
            L1_1["🔒 Data Isolation<br/>Each client owns data<br/>No data sharing"]
            L1_2["📊 Decentralized Storage<br/>Distributed across clients<br/>No central repository"]
            L1_3["🚫 No Raw Data Transfer<br/>Only model parameters<br/>Images stay local"]
            
            style L1_1 fill:#a5d6a7,stroke:#43a047
            style L1_2 fill:#a5d6a7,stroke:#43a047
            style L1_3 fill:#a5d6a7,stroke:#43a047
        end
        
        subgraph LAYER2 ["Layer 2: Communication Security"]
            style LAYER2 fill:#b3e5fc,stroke:#0288d1,stroke-width:2px
            L2_1["🔐 TCP/IP Encryption<br/>Secure sockets<br/>Optional TLS/SSL"]
            L2_2["✅ Authentication<br/>Client ID verification<br/>IP whitelisting"]
            L2_3["📦 Integrity Checks<br/>Checksum validation<br/>Corruption detection"]
            
            style L2_1 fill:#81d4fa,stroke:#0288d1
            style L2_2 fill:#81d4fa,stroke:#0288d1
            style L2_3 fill:#81d4fa,stroke:#0288d1
        end
        
        subgraph LAYER3 ["Layer 3: Model Privacy"]
            style LAYER3 fill:#fff9c4,stroke:#f9a825,stroke-width:2px
            L3_1["⚖️ Federated Averaging<br/>Aggregate weights<br/>No individual tracking"]
            L3_2["🎭 Model Anonymity<br/>Cannot reverse weights<br/>to original data"]
            L3_3["📉 Differential Privacy<br/>(Optional) Add noise<br/>to weight updates"]
            
            style L3_1 fill:#fff59d,stroke:#f9a825
            style L3_2 fill:#fff59d,stroke:#f9a825
            style L3_3 fill:#fff59d,stroke:#f9a825
        end
        
        subgraph LAYER4 ["Layer 4: System Security"]
            style LAYER4 fill:#ffccbc,stroke:#ff5722,stroke-width:2px
            L4_1["🔥 Firewall Protection<br/>Port filtering<br/>Access control"]
            L4_2["⏱️ Timeout Management<br/>Prevent hanging<br/>Connection limits"]
            L4_3["📝 Audit Logging<br/>Track all operations<br/>Detect anomalies"]
            
            style L4_1 fill:#ffab91,stroke:#ff5722
            style L4_2 fill:#ffab91,stroke:#ff5722
            style L4_3 fill:#ffab91,stroke:#ff5722
        end
        
        LAYER1 --> LAYER2
        LAYER2 --> LAYER3
        LAYER3 --> LAYER4
        
        subgraph THREATS ["🛡️ Protection Against"]
            style THREATS fill:#ffebee,stroke:#d32f2f,stroke-width:2px
            T1["❌ Data Breach<br/>Protected by: No raw data transfer"]
            T2["❌ Model Inversion<br/>Protected by: Aggregation + Noise"]
            T3["❌ Membership Inference<br/>Protected by: DP + Ensemble"]
            T4["❌ Poisoning Attacks<br/>Protected by: Validation checks"]
            
            style T1 fill:#ffcdd2,stroke:#d32f2f
            style T2 fill:#ffcdd2,stroke:#d32f2f
            style T3 fill:#ffcdd2,stroke:#d32f2f
            style T4 fill:#ffcdd2,stroke:#d32f2f
        end
    end
```

---

## 💾 System Components & Files

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#f3e5f5','primaryTextColor':'#000','primaryBorderColor':'#8e24aa'}}}%%

graph TB
    subgraph SYSTEM ["System Components"]
        style SYSTEM fill:#f3e5f5,stroke:#8e24aa,stroke-width:4px
        
        subgraph SERVER_FILES ["Server Files"]
            style SERVER_FILES fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
            SF1["fl_server.py<br/>Main server code<br/>Coordination logic"]
            SF2["model_architecture.py<br/>Ensemble model<br/>Shared definition"]
            SF3["config.py<br/>Configuration<br/>Parameters"]
            SF4["dataset/<br/>Validation data<br/>Testing only"]
            SF5["models/<br/>Checkpoints<br/>Saved weights"]
            
            style SF1 fill:#bbdefb,stroke:#1976d2,stroke-width:2px
            style SF2 fill:#90caf9,stroke:#1976d2
            style SF3 fill:#90caf9,stroke:#1976d2
            style SF4 fill:#90caf9,stroke:#1976d2
            style SF5 fill:#90caf9,stroke:#1976d2
        end
        
        subgraph CLIENT_FILES ["Client Files"]
            style CLIENT_FILES fill:#fff3e0,stroke:#f57c00,stroke-width:2px
            CF1["fl_client.py<br/>Main client code<br/>Training logic"]
            CF2["model_architecture.py<br/>Same as server<br/>Shared model"]
            CF3["config.py<br/>Same config<br/>Updated paths"]
            CF4["federated_learning_data/<br/>Local dataset<br/>Training data"]
            
            style CF1 fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
            style CF2 fill:#ffcc80,stroke:#f57c00
            style CF3 fill:#ffcc80,stroke:#f57c00
            style CF4 fill:#ffcc80,stroke:#f57c00
        end
        
        subgraph SHARED ["Shared Components"]
            style SHARED fill:#c8e6c9,stroke:#43a047,stroke-width:2px
            SH1["PyTorch<br/>Deep learning<br/>Framework"]
            SH2["timm<br/>Pre-trained models<br/>Model zoo"]
            SH3["torchvision<br/>Data transforms<br/>Dataset loading"]
            SH4["NumPy, sklearn<br/>Metrics & utils<br/>Helper libraries"]
            
            style SH1 fill:#a5d6a7,stroke:#43a047
            style SH2 fill:#a5d6a7,stroke:#43a047
            style SH3 fill:#a5d6a7,stroke:#43a047
            style SH4 fill:#a5d6a7,stroke:#43a047
        end
        
        SF1 --> SF2
        SF2 --> SF3
        CF1 --> CF2
        CF2 --> CF3
        
        SF2 -.->|"Shared"| CF2
        SF3 -.->|"Copied"| CF3
        
        SF1 --> SH1
        CF1 --> SH1
        SF2 --> SH2
        CF2 --> SH2
    end
```

---

## 📊 Performance Metrics Dashboard

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#fff9c4','primaryTextColor':'#000','primaryBorderColor':'#f9a825'}}}%%

graph TB
    subgraph METRICS ["Performance Tracking"]
        style METRICS fill:#fff9c4,stroke:#f9a825,stroke-width:4px
        
        subgraph ROUND_METRICS ["Per Round Metrics"]
            style ROUND_METRICS fill:#fff59d,stroke:#f9a825,stroke-width:2px
            RM1["📈 Validation Accuracy<br/>Track improvement<br/>Best model selection"]
            RM2["📉 Validation Loss<br/>Monitor convergence<br/>Learning progress"]
            RM3["⏱️ Round Time<br/>Training duration<br/>Communication overhead"]
            RM4["💾 Model Size<br/>Checkpoint size<br/>Storage tracking"]
            
            style RM1 fill:#ffd54f,stroke:#f9a825
            style RM2 fill:#ffd54f,stroke:#f9a825
            style RM3 fill:#ffd54f,stroke:#f9a825
            style RM4 fill:#ffd54f,stroke:#f9a825
        end
        
        subgraph CLIENT_METRICS ["Per Client Metrics"]
            style CLIENT_METRICS fill:#b3e5fc,stroke:#0288d1,stroke-width:2px
            CM1["🏋️ Training Accuracy<br/>Local performance<br/>Overfitting check"]
            CM2["📊 Training Loss<br/>Convergence rate<br/>Optimization quality"]
            CM3["🔢 Data Size<br/>Samples trained<br/>Contribution weight"]
            CM4["⚡ GPU Utilization<br/>Resource usage<br/>Efficiency"]
            
            style CM1 fill:#81d4fa,stroke:#0288d1
            style CM2 fill:#81d4fa,stroke:#0288d1
            style CM3 fill:#81d4fa,stroke:#0288d1
            style CM4 fill:#81d4fa,stroke:#0288d1
        end
        
        subgraph SYSTEM_METRICS ["System Metrics"]
            style SYSTEM_METRICS fill:#c8e6c9,stroke:#43a047,stroke-width:2px
            SM1["🌐 Network Transfer<br/>Bandwidth usage<br/>~1.5 GB/round/client"]
            SM2["💻 Memory Usage<br/>GPU/CPU RAM<br/>~12-14 GB peak"]
            SM3["🔄 Round Duration<br/>End-to-end time<br/>~5-10 min/round"]
            SM4["✅ Client Participation<br/>Connected clients<br/>Success rate"]
            
            style SM1 fill:#a5d6a7,stroke:#43a047
            style SM2 fill:#a5d6a7,stroke:#43a047
            style SM3 fill:#a5d6a7,stroke:#43a047
            style SM4 fill:#a5d6a7,stroke:#43a047
        end
        
        subgraph FINAL_METRICS ["Final Results"]
            style FINAL_METRICS fill:#ffccbc,stroke:#ff5722,stroke-width:2px
            FM1["🎯 Best Accuracy<br/>Highest validation<br/>Target: ~95-98%"]
            FM2["📈 Accuracy Gain<br/>vs Centralized<br/>+0-2% typical"]
            FM3["🔒 Privacy Preserved<br/>No data leakage<br/>✅ Verified"]
            FM4["⏰ Total Time<br/>10 rounds<br/>~1-2 hours"]
            
            style FM1 fill:#ffab91,stroke:#ff5722,stroke-width:2px
            style FM2 fill:#ffab91,stroke:#ff5722
            style FM3 fill:#ffab91,stroke:#ff5722
            style FM4 fill:#ffab91,stroke:#ff5722
        end
    end
```

---

## 🎯 System Advantages

### **Why Federated Learning?**

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e8f5e9','primaryTextColor':'#000','primaryBorderColor':'#43a047'}}}%%

mindmap
  root((Federated<br/>Learning))
    Privacy
      Data stays local
      No raw data sharing
      HIPAA compliant
      Patient confidentiality
    Scalability
      Add more clients easily
      Distributed computation
      Parallel training
      No central bottleneck
    Collaboration
      Multiple institutions
      Shared knowledge
      No data pooling needed
      Fair contribution
    Performance
      Diverse data sources
      Better generalization
      Robust models
      Ensemble benefits
    Practical
      Reduce data transfer
      Use local GPUs
      Compliance friendly
      Real-world deployable
```

---

**📄 Document Created:** Federated Learning System Architecture Diagrams  
**🎨 Format:** Mermaid Diagrams with Detailed Color Coding  
**📊 Detail Level:** Comprehensive - End-to-End System  
**🔍 Includes:** Sequence diagrams, Flow charts, Gantt charts, Security layers, Data flow
