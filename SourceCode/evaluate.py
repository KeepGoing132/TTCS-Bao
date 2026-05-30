"""
Evaluation and inference script for RAF-DB model
"""
import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

import config
from dataset import get_dataloaders
from models import get_model
from losses import get_criterion
from utils import accuracy, load_checkpoint


def evaluate_on_test_set(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy_val = correct / total if total > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy_val,
        'correct': correct,
        'total': total,
        'predictions': all_preds,
        'labels': all_labels
    }


def predict_image(image_path, model, device):
    """Predict emotion for a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.RAF_DB_MEAN,
            std=config.RAF_DB_STD
        )
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    emotion = config.EMOTION_LABELS.get(predicted_class, "Unknown")
    
    return {
        'emotion': emotion,
        'class_id': predicted_class,
        'confidence': confidence,
        'probabilities': {
            config.EMOTION_LABELS[i]: probabilities[0, i].item()
            for i in range(len(config.EMOTION_LABELS))
        }
    }


def batch_predict_images(image_dir, model, device):
    """Predict emotions for all images in a directory"""
    results = []
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(image_dir, filename)
            try:
                result = predict_image(image_path, model, device)
                result['filename'] = filename
                results.append(result)
                print(f"{filename}: {result['emotion']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return results


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        cm = confusion_matrix(y_true, y_pred)
        labels = [config.EMOTION_LABELS[i] for i in range(len(config.EMOTION_LABELS))]
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        
        os.makedirs(config.LOG_DIR, exist_ok=True)
        save_path = os.path.join(config.LOG_DIR, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=100)
        plt.close()
        
        print(f"\nConfusion matrix saved to {save_path}")
    except ImportError:
        print("Matplotlib and scikit-learn required for confusion matrix visualization")


def print_classification_report(y_true, y_pred):
    """Print classification metrics"""
    try:
        from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
        
        labels = list(range(len(config.EMOTION_LABELS)))
        label_names = [config.EMOTION_LABELS[i] for i in labels]
        
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        print(f"Weighted F1-Score: {f1:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        
    except ImportError:
        print("Scikit-learn required for detailed classification metrics")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate emotion recognition model')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to checkpoint')
    parser.add_argument('--mode', choices=['test', 'image', 'batch'], default='test',
                        help='Evaluation mode (default: test)')
    parser.add_argument('--image', type=str,
                        help='Image path for single image prediction')
    parser.add_argument('--batch-dir', type=str,
                        help='Directory for batch prediction')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU device id (default: 0)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading ResNet50 model from {args.checkpoint}...")
    model = get_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False
    )
    model, _, _, _, _ = load_checkpoint(args.checkpoint, model, None, device)
    model = model.to(device)
    
    # Evaluation mode
    if args.mode == 'test':
        print("Evaluating on test set...")
        _, _, test_loader = get_dataloaders()
        criterion = get_criterion()
        
        results = evaluate_on_test_set(model, test_loader, criterion, device)
        
        print(f"\nTest Results:")
        print(f"Loss: {results['loss']:.4f}")
        print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
        
        plot_confusion_matrix(results['labels'], results['predictions'])
        print_classification_report(results['labels'], results['predictions'])
    
    # Single image prediction
    elif args.mode == 'image':
        if not args.image:
            print("Please provide --image <path>")
            return
        
        print(f"Predicting emotion for {args.image}...")
        result = predict_image(args.image, model, device)
        
        print(f"\nPredicted emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nProbabilities:")
        for emotion, prob in result['probabilities'].items():
            print(f"  {emotion}: {prob:.4f}")
    
    # Batch prediction
    elif args.mode == 'batch':
        if not args.batch_dir:
            print("Please provide --batch-dir <path>")
            return
        
        print(f"Predicting emotions for images in {args.batch_dir}...")
        results = batch_predict_images(args.batch_dir, model, device)
        
        print(f"\nProcessed {len(results)} images")


if __name__ == '__main__':
    main()
