# Test2.py - Neural Collaborative Filtering Pipeline

## 📋 Overview

`test2.py` implements a complete recommendation system using **Neural Matrix Factorization (NeuMF)** for TV show recommendations. The pipeline includes data preprocessing, hyperparameter tuning, model training, and inference on a test set.

---

## 🔄 Pipeline Architecture

### 1. **Data Loading & Preprocessing**

- Loads training, validation, and test data from parquet files
- **Input files:**
  - `logs_train.parquet`, `logs_val.parquet` - User-item interaction logs
  - `metadata_train.parquet`, `metadata_val.parquet`, `metadata_test.parquet` - Item metadata
- **Processing steps:**
  - Filters out records with `tv_show_id == 0` (invalid items)
  - Converts user/item IDs to indices using `LabelEncoder`
  - Creates bidirectional ID mappings for later conversion

**Output:**

- `n_users`: Total unique users
- `n_items`: Total unique items
- ID mapping dictionaries

---

### 2. **Model Architecture: Neural Matrix Factorization (NeuMF)**

#### Design:

```
┌─────────────────────────────────────┐
│        User-Item Pair Input         │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
    ┌───▼────┐    ┌──▼───┐
    │  GMF   │    │ MLP  │
    │ Branch │    │Branch│
    └───┬────┘    └──┬───┘
        │             │
        └──────┬──────┘
            ┌──▼───┐
            │ FC   │
            │Layer │
            └──┬───┘
               │
            ┌──▼──┐
            │Sigmoid
            │ (0-1)
            └──────┘
```

#### Components:

- **GMF Branch:** Element-wise product of user and item embeddings
- **MLP Branch:** Concatenated embeddings through hidden layers
- **Output:** Sigmoid activation for probability score (0-1)

#### Implementation:

```python
class NeuMF(nn.Module):
    - User/Item GMF embeddings
    - User/Item MLP embeddings
    - MLP layers (default: [128, 64])
    - Dropout support
    - Xavier initialization
```

---

### 3. **Hyperparameter Tuning**

#### Search Space:

```python
{
  'alpha': [0.6, 0.7, 0.8],           # Label preference weight
  'beta': [0.2, 0.3, 0.4],            # Label preference weight
  'threshold': [0.2, 0.3, 0.4],       # Label threshold
  'lr': [0.001, 0.0005],              # Learning rate
  'batch_size': [512, 1024],          # Training batch size
  'epochs': [5, 7],                   # Training epochs
  'emb_dim': [64],                    # Embedding dimension
}
```

#### Process:

- Tests all combinations of hyperparameters (grid search)
- For each trial:
  1. Build preference labels with current hyperparameters
  2. Create negative samples (4x ratio)
  3. Train model for specified epochs
  4. Evaluate on validation set using NDCG@5
  5. Track best performing configuration

#### Evaluation Metrics:

- **Recall@5:** Hit rate among top-5 recommendations
- **MAP@5:** Mean Average Precision at 5
- **NDCG@5:** Normalized Discounted Cumulative Gain (primary metric)

---

### 4. **Training Phase**

#### Data Preparation:

1. **Aggregate interactions:** Count user-item interaction frequencies
2. **Build preference labels:**
   - Uses `alpha`, `beta`, `threshold` to assign positive/negative labels
   - Considers interaction frequency and metadata
3. **Negative sampling:** Creates 4x negative samples for each positive example

#### Training Loop:

```python
for epoch in range(epochs):
    for batch in train_loader:
        user, item, label = batch
        prediction = model(user, item)
        loss = BCELoss(prediction, label)
        loss.backward()
        optimizer.step()
```

#### Optimizer:

- **Adam optimizer** with best tuned learning rate
- **Loss function:** Binary Cross-Entropy (BCE)

---

### 5. **Inference & Predictions**

#### Test Set Preparation:

- Filters to items available in test metadata
- Maps user IDs to indices (cold-start handling for new users)

#### Prediction Strategy:

For each user:

1. **Known users:** Score all available items using trained model
2. **Cold-start users:** Return top-5 popular items
3. Select top-5 recommendations per user
4. Map item indices back to original TV show IDs

#### Output:

- **submission_final.csv** with format:
  ```
  user_id, tv_show_id
  user_001, 123 456 789 101 102
  user_002, 234 567 890 111 112
  ```

---

## 📊 Output Files

| File                                        | Purpose                          |
| ------------------------------------------- | -------------------------------- |
| `models/neumf_final_[timestamp].pth`        | Trained model weights            |
| `submission_final.csv`                      | Final predictions for submission |
| `results/loss_curve_[timestamp].png`        | Training loss visualization      |
| `results/pipeline_summary_[timestamp].json` | Pipeline statistics & config     |
| `results/tuning_results_[timestamp].csv`    | All hyperparameter trials        |

---

## 🔧 Key Functions

### `train_epoch()`

Trains model for one epoch, returns average loss.

### `evaluate_metrics()`

Computes Recall@k, MAP@k, NDCG@k on validation data.

### `predict_topk()`

Generates top-k recommendations for each user in submission set.

### `export_submission()`

Exports predictions to CSV format required for submission.

---

## 📈 Typical Results

### Data Statistics:

- **Users:** ~1000-5000
- **Items:** ~500-2000
- **Train interactions:** ~100K-500K
- **Validation interactions:** ~10K-50K

### Model Performance:

- **Best NDCG@5:** 0.25-0.45 (depends on data quality)
- **Recall@5:** 0.3-0.6
- **Training loss:** Typically converges within 5-7 epochs

---

## 🚀 How to Run

```bash
python test2.py
```

### Requirements:

```
torch
numpy
pandas
scikit-learn
matplotlib
```

### Device:

- Automatically uses CUDA if available, else CPU
- Current device printed at startup

---

## ⚠️ Validation Checks

The script validates the submission before completion:

1. ✅ Number of predictions matches submission users
2. ✅ Each user has exactly 5 recommendations
3. ✅ All recommended items exist in test metadata

---

## 🎯 Summary Output

Final summary printed includes:

- Data statistics (users, items, interactions)
- Best hyperparameters found
- Validation metrics achieved
- Final model training info
- Submission statistics

Summary also saved to `results/pipeline_summary_[timestamp].json`

---

## 💡 Key Features

✨ **Comprehensive Pipeline:** End-to-end from data loading to submission
🔍 **Hyperparameter Tuning:** Systematic grid search with metric tracking
🧠 **Advanced Model:** Combines GMF and MLP for better recommendations
⚡ **GPU Optimized:** CUDA support for faster training
❄️ **Cold Start Handling:** Fallback to popularity for new users
📊 **Detailed Logging:** Progress tracking and result visualization
