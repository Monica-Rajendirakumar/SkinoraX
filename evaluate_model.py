import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
DATASET_DIR = 'skin_dataset'
MODEL_PATH = 'skinorax_model.h5'

def load_test_data():
    """Load test dataset"""
    print("\n[1/4] Loading test data...")
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # Important: don't shuffle for evaluation
    )
    
    print(f"   ✓ Test samples: {test_generator.samples}")
    print(f"   Classes: {test_generator.class_indices}")
    
    return test_generator

def evaluate_model(model, test_gen):
    """Evaluate model on test data"""
    print("\n[2/4] Evaluating model on test set...")
    
    # Get predictions
    predictions = model.predict(test_gen, verbose=1)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    
    # Calculate metrics
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen, verbose=0)
    
    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy:  {test_acc*100:.2f}%")
    print(f"Precision: {test_precision*100:.2f}%")
    print(f"Recall:    {test_recall*100:.2f}%")
    print(f"Loss:      {test_loss:.4f}")
    
    # Calculate F1-Score
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    print(f"F1-Score:  {f1_score*100:.2f}%")
    print("=" * 60)
    
    return y_true, y_pred, predictions

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    print("\n[3/4] Generating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("   ✓ Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()
    
    return cm

def plot_roc_curve(y_true, predictions):
    """Plot ROC curve"""
    print("\n[4/4] Generating ROC curve...")
    
    fpr, tpr, thresholds = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("   ✓ ROC curve saved as 'roc_curve.png'")
    plt.close()
    
    return roc_auc

def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report"""
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("=" * 60)

def main():
    """Main evaluation function"""
    print("=" * 60)
    print("SKINORAX - MODEL EVALUATION")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model file '{MODEL_PATH}' not found!")
        print("Please run 'train_model.py' first.")
        return
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("   ✓ Model loaded successfully!")
    
    # Load test data
    test_gen = load_test_data()
    
    # Get class names
    class_names = list(test_gen.class_indices.keys())
    
    # Evaluate model
    y_true, y_pred, predictions = evaluate_model(model, test_gen)
    
    # Generate visualizations
    cm = plot_confusion_matrix(y_true, y_pred, class_names)
    roc_auc = plot_roc_curve(y_true, predictions.flatten())
    
    # Print detailed report
    print_classification_report(y_true, y_pred, class_names)
    
    # Sample predictions
    print("\nSAMPLE PREDICTIONS (First 10 images):")
    print("-" * 60)
    print(f"{'True Label':<15} {'Predicted':<15} {'Confidence':<15}")
    print("-" * 60)
    
    for i in range(min(10, len(y_true))):
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        confidence = predictions[i][0] if y_pred[i] == 1 else 1 - predictions[i][0]
        
        symbol = "✓" if y_true[i] == y_pred[i] else "✗"
        print(f"{true_label:<15} {pred_label:<15} {confidence*100:>6.2f}%  {symbol}")
    
    print("-" * 60)
    
    # Final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"✓ Confusion matrix: confusion_matrix.png")
    print(f"✓ ROC curve: roc_curve.png")
    print(f"✓ AUC Score: {roc_auc:.4f}")
    print("\nYour model is ready for deployment!")
    print("Next step: Proceed to Phase 2 (Backend Development)")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ EVALUATION ERROR: {e}")
        import traceback
        traceback.print_exc()