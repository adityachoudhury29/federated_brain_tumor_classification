#!/usr/bin/env python3
"""
Test script to verify your federated learning setup
Run this on each machine to check configuration
"""

import os
import sys
import socket

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_imports():
    """Test if all required packages are installed"""
    print_header("Testing Package Imports")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow'
    }
    
    all_good = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20} - Installed")
        except ImportError:
            print(f"✗ {name:20} - NOT INSTALLED")
            all_good = False
    
    return all_good

def test_files():
    """Test if required files exist"""
    print_header("Testing Required Files")
    
    required_files = [
        'model_architecture.py',
        'config.py'
    ]
    
    all_good = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file:30} - Found")
        else:
            print(f"✗ {file:30} - NOT FOUND")
            all_good = False
    
    return all_good

def test_config():
    """Test configuration"""
    print_header("Testing Configuration")
    
    try:
        import config
        
        print(f"Server IP:         {config.SERVER_IP}")
        print(f"Server Port:       {config.SERVER_PORT}")
        print(f"FL Rounds:         {config.NUM_FL_ROUNDS}")
        print(f"Clients Expected:  {config.NUM_CLIENTS}")
        print(f"Local Epochs:      {config.NUM_LOCAL_EPOCHS}")
        
        # Check if still using default IP
        if config.SERVER_IP == '192.168.1.100':
            print("\n⚠ WARNING: SERVER_IP is still set to default!")
            print("  Please edit config.py and set your actual server IP")
            return False
        
        return True
        
    except ImportError:
        print("✗ Cannot import config.py")
        return False
    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return False

def test_dataset(is_server=False):
    """Test if dataset exists"""
    print_header("Testing Dataset")
    
    try:
        import config
        
        if is_server:
            data_dir = config.SERVER_DATA_DIR
            print(f"Checking server dataset at: {data_dir}")
        else:
            data_dir = config.CLIENT_DATA_DIR
            print(f"Checking client dataset at: {data_dir}")
        
        train_dir = os.path.join(data_dir, 'Training')
        test_dir = os.path.join(data_dir, 'Testing')
        
        required_classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        
        all_good = True
        
        # Check Training directory
        if os.path.exists(train_dir):
            print(f"✓ Training directory found: {train_dir}")
            for cls in required_classes:
                cls_path = os.path.join(train_dir, cls)
                if os.path.exists(cls_path):
                    num_images = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  ✓ {cls:20} - {num_images} images")
                else:
                    print(f"  ✗ {cls:20} - NOT FOUND")
                    all_good = False
        else:
            print(f"✗ Training directory NOT FOUND: {train_dir}")
            all_good = False
        
        # Check Testing directory
        if os.path.exists(test_dir):
            print(f"✓ Testing directory found: {test_dir}")
        else:
            print(f"✗ Testing directory NOT FOUND: {test_dir}")
            all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"✗ Error checking dataset: {e}")
        return False

def test_network(is_server=False):
    """Test network configuration"""
    print_header("Testing Network")
    
    try:
        import config
        
        # Get local IP addresses
        hostname = socket.gethostname()
        print(f"Hostname: {hostname}")
        
        try:
            local_ips = socket.gethostbyname_ex(hostname)[2]
            print(f"Local IP addresses: {', '.join(local_ips)}")
        except:
            print("Could not determine local IP addresses")
        
        if is_server:
            # Test if we can bind to the port
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                test_socket.bind((config.SERVER_HOST, config.SERVER_PORT))
                test_socket.close()
                print(f"✓ Can bind to port {config.SERVER_PORT}")
                return True
            except Exception as e:
                print(f"✗ Cannot bind to port {config.SERVER_PORT}: {e}")
                return False
        else:
            # Test if we can reach the server
            print(f"Testing connection to server {config.SERVER_IP}:{config.SERVER_PORT}...")
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(5)
                result = test_socket.connect_ex((config.SERVER_IP, config.SERVER_PORT))
                test_socket.close()
                
                if result == 0:
                    print(f"✓ Can connect to server")
                    return True
                else:
                    print(f"✗ Cannot connect to server (make sure server is running)")
                    print(f"  Error code: {result}")
                    return False
            except Exception as e:
                print(f"✗ Network error: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Error testing network: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  FEDERATED LEARNING SETUP VERIFICATION")
    print("="*60)
    
    # Ask if server or client
    print("\nAre you testing:")
    print("1) Server machine")
    print("2) Client machine")
    
    choice = input("Enter choice (1 or 2): ").strip()
    is_server = (choice == '1')
    
    # Run tests
    results = []
    
    results.append(("Package Imports", test_imports()))
    results.append(("Required Files", test_files()))
    results.append(("Configuration", test_config()))
    results.append(("Dataset", test_dataset(is_server)))
    results.append(("Network", test_network(is_server)))
    
    # Summary
    print_header("Test Summary")
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20} - {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("  ✓ ALL TESTS PASSED")
        print("  You're ready to run federated learning!")
        if is_server:
            print("\n  Next step: Run 'python fl_server.py'")
        else:
            print("\n  Next step: Run 'python fl_client.py client_X'")
    else:
        print("  ✗ SOME TESTS FAILED")
        print("  Please fix the issues above before running FL")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
