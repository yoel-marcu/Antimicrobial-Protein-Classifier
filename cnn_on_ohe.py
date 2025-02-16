import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import os

# 🔹 **Enable GPU Acceleration**
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)  # Prevents memory allocation errors
        print("GPU memory growth set!")
    except RuntimeError as e:
        print(e)

# Set paths
DATASET_DIR = "/sci/labs/asafle/yoel.marcu2003/dataset/ohe_embeddings"
MODEL_OUTPUT = "/sci/labs/asafle/yoel.marcu2003/dataset/trained_models/cnn_model.h5"
STATS_OUTPUT = "/sci/labs/asafle/yoel.marcu2003/dataset/trained_models/training_stats.png"
os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)

# Load data:
def load_data(part):
    return np.load(os.path.join(DATASET_DIR, f"{part}_ohe.npy"))

def shuffle_data(X, y):
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

classes = ['pos_', 'neg_']
parts = ['train', 'val', 'test']

MAX_SEQ_LEN = 150
X_train_pos = pad_sequences(load_data(classes[0]+parts[0]), maxlen=MAX_SEQ_LEN, padding='post', dtype='float64')
X_val_pos = pad_sequences(load_data(classes[0]+parts[1]), maxlen=MAX_SEQ_LEN, padding='post', dtype='float64')
X_test_pos = pad_sequences(load_data(classes[0]+parts[2]), maxlen=MAX_SEQ_LEN, padding='post', dtype='float64')
X_train_neg = pad_sequences(load_data(classes[1]+parts[0]), maxlen=MAX_SEQ_LEN, padding='post', dtype='float64')
X_val_neg = pad_sequences(load_data(classes[1]+parts[1]), maxlen=MAX_SEQ_LEN, padding='post', dtype='float64')
X_test_neg = pad_sequences(load_data(classes[1]+parts[2]), maxlen=MAX_SEQ_LEN, padding='post', dtype='float64')

# Create labels
y_train = np.concatenate([np.ones(len(X_train_pos)), np.zeros(len(X_train_neg))])
y_val = np.concatenate([np.ones(len(X_val_pos)), np.zeros(len(X_val_neg))])
y_test = np.concatenate([np.ones(len(X_test_pos)), np.zeros(len(X_test_neg))])

# Combine positive and negative sets
X_train = np.concatenate([X_train_pos, X_train_neg])
X_val = np.concatenate([X_val_pos, X_val_neg])
X_test = np.concatenate([X_test_pos, X_test_neg])

X_train, y_train = shuffle_data(X_train, y_train)
X_val, y_val = shuffle_data(X_val, y_val)
X_test, y_test = shuffle_data(X_test, y_test)

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

def build_cnn(input_shape):
    model = keras.Sequential([
        layers.Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and train model
input_shape = (X_train.shape[1], X_train.shape[2])
strategy = tf.distribute.MirroredStrategy()  # **Distributes training across GPUs**
with strategy.scope():  # **Ensures model runs on GPU**
    model = build_cnn(input_shape)

# Train model with GPU acceleration
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,  # **Increase epochs for better training**
    batch_size=256,  # **Larger batch size for GPU efficiency**
    class_weight=class_weight_dict,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Generate Predictions
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# Classification Report
report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(STATS_OUTPUT.replace(".png", "_confusion_matrix.png"))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig(STATS_OUTPUT.replace(".png", "_roc_curve.png"))

# Save model
model.save(MODEL_OUTPUT)
print(f"Model saved to {MODEL_OUTPUT}")

