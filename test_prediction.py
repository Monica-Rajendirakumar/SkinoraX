import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys

# Configuration
IMG_SIZE = 224
MODEL_PATH = 'skinorax_model.h5'

# Class labels
CLASS_LABELS = {0: 'Benign', 1: 'Malignant'}

def load_and_preprocess_image(img_path):
    """Load and preprocess a single image"""
    # Load image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Expand dimensions (model expects batch)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale (same as training)
    img_array = img_array / 255.0
    
    return img_array

def predict_image(model, img_path):
    """Predict skin lesion from image"""
    print("\n" + "=" * 60)
    print("SKINORAX - IMAGE PREDICTION")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"\n❌ ERROR: Image not found at '{img_path}'")
        return
    
    print(f"\nAnalyzing image: {os.path.basename(img_path)}")
    
    # Load and preprocess
    print("Processing image...")
    img_array = load_and_preprocess_image(img_path)
    
    # Make prediction
    print("Running AI model...")
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Interpret prediction
    predicted_class = 1 if prediction > 0.5 else 0
    confidence = prediction if predicted_class == 1 else 1 - prediction
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Diagnosis:   {CLASS_LABELS[predicted_class]}")
    print(f"Confidence:  {confidence * 100:.2f}%")
    print(f"Risk Score:  {prediction * 100:.2f}%")
    print("=" * 60)
    
    # Risk interpretation
    print("\nINTERPRETATION:")
    if predicted_class == 1:
        if confidence > 0.8:
            print("⚠️  HIGH RISK - Immediate medical consultation recommended")
        elif confidence > 0.6:
            print("⚠️  MODERATE RISK - Professional evaluation advised")
        else:
            print("⚠️  LOW-MODERATE RISK - Monitor and consult if concerned")
    else:
        if confidence > 0.8:
            print("✓ LOW RISK - Appears benign, routine monitoring recommended")
        elif confidence > 0.6:
            print("✓ LIKELY BENIGN - Consider professional verification")
        else:
            print("⚠️  UNCERTAIN - Professional evaluation recommended")
    
    print("\n⚕️  DISCLAIMER: This is an AI-assisted screening tool.")
    print("   Always consult a qualified dermatologist for medical advice.")
    print("=" * 60)
    
    return predicted_class, confidence

def main():
    """Main prediction function"""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model file '{MODEL_PATH}' not found!")
        print("Please run 'train_model.py' first.")
        return
    
    # Load model
    print("Loading SkinoraX AI model...")
    model = load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Try to find a test image
        test_dir = os.path.join('skin_dataset', 'test')
        
        if os.path.exists(test_dir):
            # Find first image in test set
            for class_folder in os.listdir(test_dir):
                class_path = os.path.join(test_dir, class_folder)
                if os.path.isdir(class_path):
                    images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
                    if images:
                        img_path = os.path.join(class_path, images[0])
                        break
            else:
                print("\n❌ No test images found!")
                print("Usage: python test_prediction.py <image_path>")
                return
        else:
            print("\n❌ No test data found!")
            print("Usage: python test_prediction.py <image_path>")
            return
    
    # Make prediction
    predict_image(model, img_path)
    
    # Interactive mode
    print("\n" + "=" * 60)
    while True:
        response = input("\nPredict another image? (y/n): ").strip().lower()
        if response == 'y':
            img_path = input("Enter image path: ").strip()
            predict_image(model, img_path)
        else:
            print("\nThank you for using SkinoraX! Stay healthy! 🏥")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting... Thank you for using SkinoraX!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()