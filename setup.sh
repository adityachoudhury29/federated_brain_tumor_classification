#!/bin/bash

# Federated Learning Quick Start Script

echo "================================================"
echo "Federated Learning Setup Helper"
echo "================================================"
echo ""

# Function to get IP address
get_ip() {
    echo "Your IP addresses:"
    hostname -I
    echo ""
}

# Check if running as server or client
echo "Are you setting up the:"
echo "1) Server (your main machine)"
echo "2) Client (remote machine)"
read -p "Enter choice (1 or 2): " choice

echo ""

if [ "$choice" = "1" ]; then
    echo "=== SERVER SETUP ==="
    echo ""
    
    # Show server IP
    echo "Your server IP addresses are:"
    hostname -I
    echo ""
    
    echo "Please edit config.py and set SERVER_IP to your server's IP"
    echo ""
    
    # Check if dataset exists
    if [ -d "dataset/Training" ] && [ -d "dataset/Testing" ]; then
        echo "✓ Dataset found in dataset/ directory"
    else
        echo "⚠ Dataset not found!"
        echo "Please ensure your dataset is in:"
        echo "  ./dataset/Training/"
        echo "  ./dataset/Testing/"
    fi
    echo ""
    
    # Create models directory
    mkdir -p models
    echo "✓ Created models/ directory for saving checkpoints"
    echo ""
    
    echo "To start the server, run:"
    echo "  python fl_server.py"
    echo ""

elif [ "$choice" = "2" ]; then
    echo "=== CLIENT SETUP ==="
    echo ""
    
    read -p "Enter the server's IP address: " server_ip
    echo ""
    
    echo "Please:"
    echo "1. Copy these files from server to this machine:"
    echo "   - model_architecture.py"
    echo "   - config.py"
    echo "   - fl_client.py"
    echo ""
    
    echo "2. Edit config.py and set:"
    echo "   SERVER_IP = '$server_ip'"
    echo "   CLIENT_DATA_DIR = '<path_to_your_dataset>'"
    echo ""
    
    echo "3. Create dataset directory:"
    read -p "Enter path for client dataset (e.g., ~/federated_learning_data): " data_path
    mkdir -p "$data_path"/{Training,Testing}
    echo "   ✓ Created $data_path/Training/"
    echo "   ✓ Created $data_path/Testing/"
    echo ""
    echo "   Please place your dataset in this structure:"
    echo "   $data_path/"
    echo "   ├── Training/"
    echo "   │   ├── glioma_tumor/"
    echo "   │   ├── meningioma_tumor/"
    echo "   │   ├── no_tumor/"
    echo "   │   └── pituitary_tumor/"
    echo "   └── Testing/"
    echo "       ├── glioma_tumor/"
    echo "       ├── meningioma_tumor/"
    echo "       ├── no_tumor/"
    echo "       └── pituitary_tumor/"
    echo ""
    
    echo "4. To start the client, run:"
    echo "   python fl_client.py client_1"
    echo ""
    
else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo "================================================"
echo "For detailed instructions, see README.md"
echo "================================================"
