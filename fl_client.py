"""
Federated Learning Client
Run this on each client machine with local dataset
"""

import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import os
import copy

from model_architecture import build_model
import config


class FederatedClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Client {client_id} using device: {self.device}")
        
        # Load local dataset
        self.load_local_data()
        
        # Initialize local model (Ensemble)
        print("\nInitializing local ensemble model...")
        self.local_model = build_model(
            num_classes=config.NUM_CLASSES, 
            pretrained=True, 
            device=self.device.type
        ).to(self.device)
        print("✓ Local ensemble model initialized")
    
    def load_local_data(self):
        """Load client's local dataset"""
        print(f"\nLoading local dataset for Client {self.client_id}...")
        
        data_dir = config.CLIENT_DATA_DIR
        train_dir = os.path.join(data_dir, config.CLIENT_TRAIN_DIR)
        test_dir = os.path.join(data_dir, config.CLIENT_TEST_DIR)
        
        # Check if dataset exists
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            raise FileNotFoundError(
                f"Dataset not found at {data_dir}\n"
                f"Please ensure the dataset is placed in:\n"
                f"  {train_dir}\n"
                f"  {test_dir}\n"
                f"See README for dataset structure."
            )
        
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=train_transform)
        
        # Combine for training
        self.train_dataset = ConcatDataset([train_dataset, test_dataset])
        self.class_names = train_dataset.classes
        
        print(f"Loaded {len(self.train_dataset)} training images")
        print(f"Classes: {self.class_names}")
        
        # Create data loader
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=2
        )
    
    def connect_to_server(self, fl_round):
        """Connect to server for one federated learning round"""
        print(f"\n{'='*60}")
        print(f"Client {self.client_id} - FL Round {fl_round + 1}")
        print(f"{'='*60}")
        
        try:
            # Connect to server
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((config.SERVER_IP, config.SERVER_PORT))
            print(f"Connected to server at {config.SERVER_IP}:{config.SERVER_PORT}")
            
            # Receive global model weights
            self.receive_model(client_socket)
            
            # Train on local data
            self.train_local_model()
            
            # Send updated weights back to server
            self.send_weights(client_socket)
            
            client_socket.close()
            print(f"Disconnected from server")
            
            return True
            
        except Exception as e:
            print(f"Error connecting to server: {e}")
            print(f"Make sure the server is running at {config.SERVER_IP}:{config.SERVER_PORT}")
            return False
    
    def receive_model(self, client_socket):
        """Receive global model weights from server"""
        print("\nReceiving global model from server...")
        
        # Receive size
        size_data = client_socket.recv(8)
        size = int.from_bytes(size_data, byteorder='big')
        
        # Receive weights data
        data = b''
        while len(data) < size:
            packet = client_socket.recv(min(size - len(data), config.BUFFER_SIZE))
            if not packet:
                break
            data += packet
        
        # Deserialize and load weights
        global_weights = pickle.loads(data)
        self.local_model.load_state_dict(global_weights)
        print("✓ Global model loaded successfully")
    
    def train_local_model(self):
        """Train model on local data"""
        print(f"\nTraining on local data for {config.NUM_LOCAL_EPOCHS} epochs...")
        
        self.local_model.train()
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Add label smoothing
        optimizer = optim.AdamW(
            self.local_model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=1e-4
        )
        
        for epoch in range(config.NUM_LOCAL_EPOCHS):
            running_loss = 0.0
            running_corrects = 0
            total = 0
            
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.local_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += inputs.size(0)
            
            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total
            
            print(f"  Epoch {epoch + 1}/{config.NUM_LOCAL_EPOCHS} - "
                  f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        print("✓ Local training completed")
    
    def send_weights(self, client_socket):
        """Send trained weights to server"""
        print("\nSending trained weights to server...")
        
        # Prepare data to send
        weights = self.local_model.state_dict()
        data_to_send = {
            'weights': weights,
            'data_size': len(self.train_dataset),
            'client_id': self.client_id
        }
        
        data = pickle.dumps(data_to_send)
        
        # Send size first
        size = len(data)
        client_socket.sendall(size.to_bytes(8, byteorder='big'))
        
        # Send data
        client_socket.sendall(data)
        print("✓ Weights sent successfully")
    
    def run(self):
        """Run federated learning for all rounds"""
        print(f"\n{'='*60}")
        print(f"Starting Federated Learning Client {self.client_id}")
        print(f"{'='*60}")
        print(f"Server: {config.SERVER_IP}:{config.SERVER_PORT}")
        print(f"Local dataset size: {len(self.train_dataset)} samples")
        print(f"FL Rounds: {config.NUM_FL_ROUNDS}")
        print(f"Local epochs per round: {config.NUM_LOCAL_EPOCHS}")
        print(f"{'='*60}\n")
        
        for fl_round in range(config.NUM_FL_ROUNDS):
            success = self.connect_to_server(fl_round)
            
            if not success:
                print(f"\nFailed to complete round {fl_round + 1}. Retrying in next round...")
                continue
        
        print(f"\n{'='*60}")
        print(f"Client {self.client_id} - FEDERATED LEARNING COMPLETED")
        print(f"{'='*60}\n")


def main():
    import sys
    
    # Get client ID from command line or use default
    if len(sys.argv) > 1:
        client_id = sys.argv[1]
    else:
        client_id = input("Enter Client ID (e.g., client_1): ")
    
    print(f"Initializing Federated Learning Client: {client_id}")
    
    try:
        client = FederatedClient(client_id)
        client.run()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease set up your dataset before running the client.")
        print(f"Edit config.py and set CLIENT_DATA_DIR to your dataset location.")
    except KeyboardInterrupt:
        print("\n\nClient interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
