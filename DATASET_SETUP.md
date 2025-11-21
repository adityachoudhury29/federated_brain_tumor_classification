# Dataset Setup Guide for Federated Learning

## 📊 Dataset Structure

Your dataset must follow this **exact** structure on all machines:

```
dataset_folder/
├── Training/
│   ├── glioma_tumor/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── meningioma_tumor/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── no_tumor/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── pituitary_tumor/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── Testing/
    ├── glioma_tumor/
    │   └── ...
    ├── meningioma_tumor/
    │   └── ...
    ├── no_tumor/
    │   └── ...
    └── pituitary_tumor/
        └── ...
```

## 🔑 Important Notes

### Folder Names Must Match Exactly
- `Training` (capital T)
- `Testing` (capital T)
- `glioma_tumor` (lowercase with underscore)
- `meningioma_tumor` (lowercase with underscore)
- `no_tumor` (lowercase with underscore)
- `pituitary_tumor` (lowercase with underscore)

### Supported Image Formats
- `.jpg`
- `.jpeg`
- `.png`

## 📍 Dataset Locations

### Server Machine (Your Machine)

**Location:**
```
/home/aditya/Desktop/Everything/federated_learning/dataset/
```

**Purpose:**
- Used for **validation/testing** the global model after each FL round
- Server evaluates the aggregated model on this data
- Does NOT participate in training

**Configuration in config.py:**
```python
SERVER_DATA_DIR = '/home/aditya/Desktop/Everything/federated_learning/dataset'
```

### Client Machines

**Recommended Location:**
```
/home/username/federated_learning_data/
```

**Purpose:**
- Used for **local training** on each client
- Data stays on client machine (never sent to server)
- Each client trains on its own local data

**Configuration in config.py (on each client):**
```python
CLIENT_DATA_DIR = '/home/username/federated_learning_data'
```

**⚠️ Important:** Change `username` to actual username on each client machine!

## 🔄 Dataset Distribution Strategies

### Strategy 1: Identical Datasets (Simple Testing)
Each client has the **same complete dataset**
- ✅ Easy to set up
- ✅ Good for testing FL setup
- ❌ Not realistic FL scenario
- ❌ No privacy benefit

### Strategy 2: Disjoint Split (Realistic FL)
Split your data across clients (each client gets different subset)
- ✅ Realistic federated learning
- ✅ True privacy benefit
- ✅ Simulates real-world scenario
- ⚠️ Need to manually split data

**Example Split for 3 Clients:**
```
Total: 3000 images of each class

Client 1: images 1-1000 of each class
Client 2: images 1001-2000 of each class
Client 3: images 2001-3000 of each class
```

### Strategy 3: Non-IID Split (Advanced)
Each client specializes in certain tumor types
- Client 1: 80% glioma, 20% others
- Client 2: 80% meningioma, 20% others
- Client 3: 80% pituitary, 20% others

## 🛠️ How to Set Up Datasets

### Option A: Using Your Existing Dataset

If you already have the dataset on your server:

#### 1. For Server (already done):
```bash
# Already at correct location
/home/aditya/Desktop/Everything/federated_learning/dataset/
```

#### 2. For Each Client:

**Method 1: Copy entire dataset (for testing)**
```bash
# On client machine
mkdir -p ~/federated_learning_data

# Copy from server (if accessible via network)
scp -r aditya@server_ip:/home/aditya/Desktop/Everything/federated_learning/dataset/* ~/federated_learning_data/
```

**Method 2: Split dataset (for realistic FL)**
```bash
# On server, create splits for each client
python3 << 'EOF'
import os
import shutil
from pathlib import Path

source = Path('/home/aditya/Desktop/Everything/federated_learning/dataset')
output_base = Path('/tmp/client_datasets')

classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
splits = ['Training', 'Testing']
num_clients = 3

for client_id in range(1, num_clients + 1):
    for split in splits:
        for cls in classes:
            source_dir = source / split / cls
            dest_dir = output_base / f'client_{client_id}' / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            images = sorted(list(source_dir.glob('*.jpg')))
            # Distribute images to clients
            client_images = images[client_id-1::num_clients]
            
            for img in client_images:
                shutil.copy2(img, dest_dir)
            
            print(f"Client {client_id}, {split}/{cls}: {len(client_images)} images")

print("Datasets created in /tmp/client_datasets/")
EOF

# Then copy to each client
# For client 1:
scp -r /tmp/client_datasets/client_1/* client1_user@client1_ip:~/federated_learning_data/
```

### Option B: Download Dataset on Each Machine

If clients will download their own datasets:

```bash
# On each client machine
cd ~
mkdir -p federated_learning_data
cd federated_learning_data

# Download your dataset (example with kaggle)
kaggle datasets download -d <dataset-name>
unzip dataset.zip
# Ensure structure matches required format
```

## ✅ Verification

After setting up dataset on any machine, verify:

```bash
# Check structure
tree -d -L 3 /path/to/your/dataset/

# Count images in each class
echo "Training - Glioma:"
ls /path/to/dataset/Training/glioma_tumor/ | wc -l
echo "Training - Meningioma:"
ls /path/to/dataset/Training/meningioma_tumor/ | wc -l
# ... repeat for all classes

# Or use the test script
python test_setup.py
```

## 📏 Recommended Dataset Sizes

### Minimum for Testing:
- Training: ~100 images per class per client
- Testing: ~50 images per class per client
- Total per client: ~600 images

### Recommended for Training:
- Training: ~500-1000 images per class per client
- Testing: ~200-300 images per class per client
- Total per client: ~2800-5200 images

### For Production:
- Training: 1000+ images per class per client
- Testing: 300+ images per class per client
- Total per client: 5200+ images

## 🔍 Common Issues

### Issue: "Dataset not found"
**Solution:**
1. Check path in config.py matches actual location
2. Verify folder names match exactly (case-sensitive!)
3. Run: `python test_setup.py`

### Issue: "No images found in dataset"
**Solution:**
1. Ensure images have correct extensions (.jpg, .jpeg, .png)
2. Check images are in correct subdirectories
3. Verify images are not corrupted

### Issue: Different class names
**Solution:**
- Rename folders to exact names: `glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor`
- Or update class names in code (not recommended)

## 💡 Tips

1. **Start small:** Test with 1 client and small dataset first
2. **Balanced classes:** Try to have similar number of images per class
3. **Quality over quantity:** Better to have fewer high-quality images
4. **Backup:** Keep original dataset backed up before splitting
5. **Document splits:** Keep track of which images went to which client

## 📝 Dataset Configuration Checklist

**On Server:**
- [ ] Dataset at `/home/aditya/Desktop/Everything/federated_learning/dataset/`
- [ ] Has Training/ and Testing/ folders
- [ ] All 4 tumor classes present
- [ ] Images are valid and loadable
- [ ] config.py has correct SERVER_DATA_DIR

**On Each Client:**
- [ ] Dataset at configured CLIENT_DATA_DIR location
- [ ] Has Training/ and Testing/ folders
- [ ] All 4 tumor classes present
- [ ] Images are valid and loadable
- [ ] config.py has correct CLIENT_DATA_DIR
- [ ] Sufficient disk space for dataset

---

**For setup help, run:** `python test_setup.py`
