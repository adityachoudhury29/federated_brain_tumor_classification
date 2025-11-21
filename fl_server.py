"""
Federated Learning Server
Run this on your main server machine
"""

import socket
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import os
import copy
import numpy as np
from datetime import datetime

from model_architecture import MobileNetV2
import config


class FederatedServer:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Server using device: {self.device}")
        
        # Initialize global model
        self.global_model = MobileNetV2(num_classes=config.NUM_CLASSES).to(self.device)
        print("Global model initialized")
        
        # Create model save directory
        os.makedirs(config.SERVER_MODEL_SAVE_PATH, exist_ok=True)
        
        # Load server's validation dataset
        self.load_validation_data()
        
        # Track training history
        self.history = {
            'rounds': [],
            'val_accuracy': [],
            'val_loss': []
        }
    
    def load_validation_data(self):
        """Load validation/test dataset on server for evaluation"""
        print("\nLoading validation dataset on server...")
        
        data_dir = config.SERVER_DATA_DIR
        train_dir = os.path.join(data_dir, config.SERVER_TRAIN_DIR)
        test_dir = os.path.join(data_dir, config.SERVER_TEST_DIR)
        
        # Transforms
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            train_dataset = datasets.ImageFolder(train_dir, transform=test_transform)
            test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
            self.val_dataset = ConcatDataset([train_dataset, test_dataset])
            self.class_names = train_dataset.classes
            print(f"Loaded {len(self.val_dataset)} validation images")
            print(f"Classes: {self.class_names}")
        else:
            print("Warning: Validation dataset not found on server")
            self.val_dataset = None
            self.class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    
    def start_server(self):
        """Start the federated learning server"""
        print(f"\n{'='*60}")
        print(f"Starting Federated Learning Server")
        print(f"{'='*60}")
        print(f"Server listening on {config.SERVER_HOST}:{config.SERVER_PORT}")
        print(f"Expected clients: {config.NUM_CLIENTS}")
        print(f"Minimum clients to start: {config.MIN_CLIENTS}")
        print(f"FL Rounds: {config.NUM_FL_ROUNDS}")
        print(f"Local epochs per round: {config.NUM_LOCAL_EPOCHS}")
        print(f"{'='*60}\n")
        
        # Run federated learning rounds
        for fl_round in range(config.NUM_FL_ROUNDS):
            print(f"\n{'='*60}")
            print(f"FEDERATED LEARNING ROUND {fl_round + 1}/{config.NUM_FL_ROUNDS}")
            print(f"{'='*60}")
            
            # Wait for clients and collect their trained weights
            client_weights = self.coordinate_round(fl_round)
            
            if len(client_weights) >= config.MIN_CLIENTS:
                # Aggregate weights using FedAvg
                self.aggregate_weights(client_weights)
                
                # Evaluate global model
                if self.val_dataset is not None:
                    val_acc, val_loss = self.evaluate_global_model()
                    self.history['rounds'].append(fl_round + 1)
                    self.history['val_accuracy'].append(val_acc)
                    self.history['val_loss'].append(val_loss)
                    print(f"\n[Round {fl_round + 1}] Global Model - Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save model checkpoint
                self.save_checkpoint(fl_round)
            else:
                print(f"\n[Warning] Not enough clients ({len(client_weights)}/{config.MIN_CLIENTS}). Skipping round.")
        
        print(f"\n{'='*60}")
        print("FEDERATED LEARNING COMPLETED")
        print(f"{'='*60}")
        self.save_final_model()
        self.print_summary()
    
    def coordinate_round(self, fl_round):
        """Coordinate one federated learning round"""
        client_weights = []
        client_data_sizes = []
        
        # Create socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((config.SERVER_HOST, config.SERVER_PORT))
        server_socket.listen(config.NUM_CLIENTS)
        server_socket.settimeout(config.TIMEOUT)
        
        print(f"\nWaiting for clients to connect...")
        
        connected_clients = 0
        while connected_clients < config.NUM_CLIENTS:
            try:
                client_socket, client_address = server_socket.accept()
                print(f"Client {connected_clients + 1} connected from {client_address}")
                
                # Send global model weights to client
                self.send_model(client_socket)
                
                # Receive trained model weights from client
                weights, data_size = self.receive_weights(client_socket)
                
                if weights is not None:
                    client_weights.append(weights)
                    client_data_sizes.append(data_size)
                    print(f"Received weights from client {connected_clients + 1} (trained on {data_size} samples)")
                
                client_socket.close()
                connected_clients += 1
                
                # If we have minimum clients, we can proceed
                if connected_clients >= config.MIN_CLIENTS and connected_clients == config.NUM_CLIENTS:
                    break
                    
            except socket.timeout:
                print(f"Timeout waiting for clients. Got {connected_clients}/{config.NUM_CLIENTS}")
                break
            except Exception as e:
                print(f"Error with client connection: {e}")
        
        server_socket.close()
        print(f"\nRound complete. Collected weights from {len(client_weights)} clients.")
        return client_weights
    
    def send_model(self, client_socket):
        """Send global model weights to client"""
        global_weights = self.global_model.state_dict()
        data = pickle.dumps(global_weights)
        
        # Send size first
        size = len(data)
        client_socket.sendall(size.to_bytes(8, byteorder='big'))
        
        # Send data
        client_socket.sendall(data)
        print("  → Sent global model to client")
    
    def receive_weights(self, client_socket):
        """Receive trained weights from client"""
        try:
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
            
            # Deserialize
            received_data = pickle.loads(data)
            weights = received_data['weights']
            data_size = received_data['data_size']
            
            print("  ← Received trained weights from client")
            return weights, data_size
        except Exception as e:
            print(f"  ✗ Error receiving weights: {e}")
            return None, 0
    
    def aggregate_weights(self, client_weights):
        """Aggregate client weights using Federated Averaging (FedAvg)"""
        print("\nAggregating weights using FedAvg...")
        
        # Initialize averaged weights
        averaged_weights = copy.deepcopy(client_weights[0])
        
        # Average all weights
        for key in averaged_weights.keys():
            for i in range(1, len(client_weights)):
                averaged_weights[key] += client_weights[i][key]
            averaged_weights[key] = torch.div(averaged_weights[key], len(client_weights))
        
        # Update global model
        self.global_model.load_state_dict(averaged_weights)
        print("Global model updated with aggregated weights")
    
    def evaluate_global_model(self):
        """Evaluate global model on server's validation set"""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        val_loader = DataLoader(self.val_dataset, batch_size=config.BATCH_SIZE, 
                               shuffle=False, num_workers=2)
        
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += inputs.size(0)
        
        val_loss = running_loss / total
        val_acc = running_corrects.double() / total
        
        return val_acc.item(), val_loss
    
    def save_checkpoint(self, fl_round):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            config.SERVER_MODEL_SAVE_PATH, 
            f'global_model_round_{fl_round + 1}.pth'
        )
        torch.save(self.global_model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model"""
        final_path = os.path.join(
            config.SERVER_MODEL_SAVE_PATH, 
            'final_global_model.pth'
        )
        torch.save(self.global_model.state_dict(), final_path)
        print(f"\nFinal model saved: {final_path}")
    
    def print_summary(self):
        """Print training summary"""
        print("\nTraining Summary:")
        print("-" * 60)
        for i, (round_num, acc, loss) in enumerate(zip(
            self.history['rounds'], 
            self.history['val_accuracy'], 
            self.history['val_loss']
        )):
            print(f"Round {round_num}: Val Acc = {acc:.4f}, Val Loss = {loss:.4f}")
        
        if self.history['val_accuracy']:
            best_acc = max(self.history['val_accuracy'])
            best_round = self.history['rounds'][self.history['val_accuracy'].index(best_acc)]
            print(f"\nBest Validation Accuracy: {best_acc:.4f} (Round {best_round})")


def main():
    print("Initializing Federated Learning Server...")
    server = FederatedServer()
    server.start_server()


if __name__ == "__main__":
    main()
