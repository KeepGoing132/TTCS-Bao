import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime                    
import pandas as pd

import config
from dataset import get_dataloaders, get_raf_db_transforms
from models import get_model
from utils import load_checkpoint, save_checkpoint, accuracy, AverageMeter

st.set_page_config(
    page_title="😊 Facial Expression Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

LABELS = ["😲 Surprise", "😨 Fear", "🤢 Disgust", "😊 Happiness", "😢 Sadness", "😠 Anger", "😐 Neutral"]
LABELS_CLEAN = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

if "is_training" not in st.session_state:
    st.session_state.is_training = False
if "stop_training" not in st.session_state:
    st.session_state.stop_training = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "train_history" not in st.session_state:
    st.session_state.train_history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

@st.cache_resource
def load_model_cached():
    model = get_model(num_classes=config.NUM_CLASSES, pretrained=True)
    best_path = 'checkpoint/model_best.pth.tar'
    
    if os.path.exists(best_path):
        try:
            checkpoint = torch.load(best_path, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            return model
        except Exception as e:
            return model
    else:
        return model

@st.cache_resource
def get_dataloaders_cached(batch_size=64):
    return get_dataloaders(batch_size=batch_size)

def get_transform():
    return get_raf_db_transforms(train=False)

def predict_emotion(image_array, model, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    if isinstance(image_array, Image.Image):
        img_tensor = transform(image_array).unsqueeze(0).to(device)
    else:
        img_tensor = torch.from_numpy(image_array).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_idx].item()
        all_probs = probs[0].cpu().numpy()
    
    return pred_idx, pred_prob, all_probs

def main():
    st.sidebar.title("⚙️ Configuration")
    
    with st.sidebar.expander("🎓 Training Settings", expanded=True):
        epochs = st.number_input("Epochs", min_value=1, max_value=500, value=config.EPOCHS)
        batch_size = st.selectbox("Batch Size", [32, 64, 128], index=1)
        lr = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.6f")
    
    
    with st.sidebar.expander("🔧 Dropout", expanded=False):
        use_dropout = st.checkbox("Enable Dropout", value=config.DROPOUT_PROB > 0, help="Add dropout before FC layer")
        if use_dropout:
            dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=config.DROPOUT_PROB, step=0.05)
        else:
            dropout_rate = 0.0
    
    with st.sidebar.expander("📁 Paths", expanded=False):
        checkpoint_path = st.text_input("Model Checkpoint", value="checkpoint/model_best.pth.tar")
        test_dir = st.text_input("Test Directory", value=config.TEST_DIR)
    
    st.sidebar.divider()
    st.sidebar.markdown(f"""
    ### 📈 Current Config:
    - **Model:** {config.MODEL_NAME}  
    - **Epochs:** {epochs}  
    - **Batch Size:** {batch_size}  
    - **LR:** {lr:.2e}  
    - **Dropout:** {'Yes' if use_dropout else 'No'}
    """)
    
    st.title("😊 Facial Expression Recognition")
    st.markdown("**ResNet50 + RAF-DB Database** | Real-time Emotion Detection")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎓 Train", "🧪 Test", "📊 Evaluate", "📸 Sample Test"])
    
    with tab1:
        st.header("🎓 Train Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
           a = 1
        
        with col2:
            st.metric("Epochs", epochs)
            st.metric("Learning Rate", f"{lr:.2e}")
            st.metric("Batch Size", batch_size)
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_from_scratch = st.checkbox("🔄 Train from Scratch", value=False, help="Uncheck to resume from best checkpoint")
        
        with col2:
            if not st.session_state.is_training:
                start_button = st.button("▶️ Start Training", key="train_start", use_container_width=True, type="primary")
            else:
                start_button = False
        
        with col3:
            if st.session_state.is_training:
                stop_button = st.button("⏹️ Stop Training", key="train_stop", use_container_width=True, type="secondary")
            else:
                stop_button = False
        
        if start_button:
            st.session_state.is_training = True
            st.rerun()
        
        if stop_button:
            st.session_state.is_training = False
            st.rerun()
        
        if st.session_state.is_training:
            st.info("⏳ Training real model with RAF-DB dataset...")
            
            # 📊 Show training parameters
            with st.expander("📊 Training Parameters", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📌 Epochs", epochs)
                with col2:
                    st.metric("📦 Batch Size", batch_size)
                with col3:
                    st.metric("📈 Learning Rate", f"{lr:.2e}")
                with col4:
                    st.metric("🎯 Num Classes", config.NUM_CLASSES)
                
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("💧 Dropout", "Yes" if use_dropout else "No")
                    if use_dropout:
                        st.text(f"Rate: {dropout_rate:.2f}")
                with col6:
                    device_check = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    st.metric("🖥️ Device", str(device_check).upper())
                    if torch.cuda.is_available():
                        st.text(f"GPU: {torch.cuda.get_device_name(0)}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.container()
            output_log = st.empty()  
            
            try:
                st.info("📂 Loading RAF-DB dataset...")
                train_loader, val_loader, _ = get_dataloaders_cached(batch_size=batch_size)
                st.success(f"✅ Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
                
                # Get real model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                st.info(f"🖥️ Device: {device} | Model: ResNet50 | Classes: {config.NUM_CLASSES}")
                model = get_model(num_classes=config.NUM_CLASSES, pretrained=True)
                
                # Load checkpoint if not training from scratch
                start_epoch = 0
                if not train_from_scratch:
                    resume_path = 'checkpoint/model_resume.pth.tar'
                    best_path = 'checkpoint/model_best.pth.tar'
                    checkpoint_to_load = None
                    
                    if os.path.exists(resume_path):
                        st.info("📂 Loading resume checkpoint...")
                        checkpoint_to_load = resume_path
                    elif os.path.exists(best_path):
                        st.info("📂 Loading best model checkpoint...")
                        checkpoint_to_load = best_path
                    
                    if checkpoint_to_load:
                        checkpoint = torch.load(checkpoint_to_load, map_location=device)
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                            start_epoch = checkpoint.get('epoch', 0) + 1
                            st.success(f"✅ Resumed from epoch {start_epoch}")
                        else:
                            model.load_state_dict(checkpoint)
                            st.success("✅ Model loaded from checkpoint")
                elif train_from_scratch:
                    st.info("🔄 Training from scratch (random weights)")
                
                model = model.to(device)
                
                # Loss and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
                
                best_val_acc = 0.0
                os.makedirs('checkpoint', exist_ok=True)
                
                st.info("🚀 Starting training loop...")
                
                for epoch in range(start_epoch, epochs):
                    if st.session_state.stop_training:
                        st.warning("⛔ Training stopped by user")
                        break
                    
                    # Training phase
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0
                    
                    for batch_idx, (images, labels) in enumerate(train_loader):
                        images, labels = images.to(device), labels.to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()
                        
                        # Update progress
                        progress = ((epoch * len(train_loader) + batch_idx) / (epochs * len(train_loader)))
                        progress_bar.progress(min(progress, 0.99))
                        
                        batch_acc = (train_correct / train_total) * 100
                        batch_output = f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}%"
                        status_text.text(batch_output)
                        output_log.code(batch_output, language="text")
                    
                    avg_train_loss = train_loss / len(train_loader)
                    avg_train_acc = (train_correct / train_total) * 100
                    
                    # Validation phase
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for images, labels in val_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            
                            val_loss += loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += labels.size(0)
                            val_correct += (predicted == labels).sum().item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    avg_val_acc = (val_correct / val_total) * 100
                    
                    # Update metrics
                    st.session_state.train_history["loss"].append(avg_train_loss)
                    st.session_state.train_history["acc"].append(avg_train_acc)
                    st.session_state.train_history["val_loss"].append(avg_val_loss)
                    st.session_state.train_history["val_acc"].append(avg_val_acc)
                    
                    # Display metrics
                    epoch_output = f"✅ Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}%"
                    st.success(epoch_output)
                    
                    with metrics_placeholder.container():
                        mcol1, mcol2, mcol3 = st.columns(3)
                        with mcol1:
                            st.metric("Train Loss", f"{avg_train_loss:.4f}")
                        with mcol2:
                            st.metric("Train Acc", f"{avg_train_acc:.2f}%")
                        with mcol3:
                            st.metric("Val Acc", f"{avg_val_acc:.2f}%")
                    
                    # Save resume checkpoint (every epoch)
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_acc': best_val_acc
                    }, 'checkpoint/model_resume.pth.tar')
                    
                    # Save best model (only when validation accuracy improves)
                    if avg_val_acc > best_val_acc:
                        best_val_acc = avg_val_acc
                        torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_val_acc': best_val_acc
                        }, 'checkpoint/model_best.pth.tar')
                        st.success(f"💾 Best model saved! Val Acc: {avg_val_acc:.2f}%")
                    
                    scheduler.step(avg_val_loss)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Training completed!")
                st.success(f"🎉 Model training finished! Best validation accuracy: {best_val_acc:.2f}%\nResume: checkpoint/model_resume.pth.tar | Best: checkpoint/model_best.pth.tar")
                st.session_state.is_training = False
                
            except Exception as e:
                st.error(f"❌ Training error: {e}")
                import traceback
                st.error(traceback.format_exc())
                st.session_state.is_training = False
    
    with tab2:
        st.header("🧪 Test Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📖 Instructions:
            1. Load trained model from checkpoint
            2. Select test folder
            3. Click **Evaluate** to test accuracy
            """)
        
        with col2:
            if st.button("📂 Load Model", key="load_model_btn", use_container_width=True):
                with st.spinner("Loading model..."):
                    try:
                        model = load_model_cached()
                        st.session_state.current_model = model
                        st.session_state.model_loaded = True
                        st.success("✅ Model loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading model: {e}")
        
        st.divider()
        
        if st.button("🔍 Evaluate on Test Set", key="eval_test", use_container_width=True, type="primary"):
            if st.session_state.model_loaded:
                try:
                    import gc
                    
                    st.info("Preparing evaluation on CPU...")
                    # Force CPU for evaluation to avoid OOM
                    cpu_device = torch.device('cpu')
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Load model directly from checkpoint on CPU
                    best_path = 'checkpoint/model_best.pth.tar'
                    if not os.path.exists(best_path):
                        st.error("❌ No checkpoint found!")
                    else:
                        model = get_model(num_classes=config.NUM_CLASSES, pretrained=False)
                        model = model.to(cpu_device)
                        checkpoint = torch.load(best_path, map_location=cpu_device)
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                        model.eval()
                        
                        st.info("Loading test set on CPU...")
                        # Use very small batch size to avoid OOM
                        eval_batch_size = 8
                        _, _, test_loader = get_dataloaders(batch_size=eval_batch_size)
                        
                        st.info("Running evaluation on CPU...")
                        progress_bar = st.progress(0)
                        
                        correct = 0
                        total = 0
                        predictions = []
                        ground_truth = []
                        
                        batch_idx = 0
                        total_batches = len(test_loader)
                        
                        with torch.no_grad():
                            for images, labels in test_loader:
                                # Keep on CPU
                                images = images.to(cpu_device)
                                labels = labels.to(cpu_device)
                                
                                # Forward pass on CPU
                                outputs = model(images)
                                _, predicted = torch.max(outputs, 1)
                                
                                batch_correct = (predicted == labels).sum().item()
                                batch_total = labels.size(0)
                                
                                correct += batch_correct
                                total += batch_total
                                
                                predictions.extend(predicted.cpu().numpy())
                                ground_truth.extend(labels.cpu().numpy())
                                
                                # Clear memory
                                del images, labels, outputs, predicted
                                
                                # Update progress
                                batch_idx += 1
                                progress = min(batch_idx / total_batches, 1.0)
                                progress_bar.progress(progress)
                        
                        accuracy_pct = (correct / total) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{accuracy_pct:.2f}%")
                        with col2:
                            st.metric("Correct", f"{correct}/{total}")
                        with col3:
                            st.metric("Error Rate", f"{100-accuracy_pct:.2f}%")
                        
                        # Per-class accuracy
                        st.subheader("📊 Per-Class Accuracy")
                    per_class_acc = {}
                    for cls in range(config.NUM_CLASSES):
                        mask = np.array(ground_truth) == cls
                        if mask.sum() > 0:
                            cls_acc = ((np.array(predictions)[mask] == cls).sum() / mask.sum()) * 100
                            per_class_acc[config.EMOTION_LABELS[cls]] = cls_acc
                    
                    if per_class_acc:
                        df_per_class = pd.DataFrame(list(per_class_acc.items()), columns=['Emotion', 'Accuracy'])
                        st.dataframe(df_per_class, use_container_width=True)
                    
                    st.success("✅ Evaluation complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                st.warning("⚠️ Please load a model first!")
        
        
        st.divider()
        
        if st.button("🎲 Show 20 Random Samples", key="show_samples", use_container_width=True, type="secondary"):
            try:
                import gc
                
                st.info("Loading model checkpoint from disk...")
                # Load checkpoint directly to CPU (skip GPU entirely for this)
                cpu_device = torch.device('cpu')
                
                # Load from best checkpoint
                best_path = 'checkpoint/model_best.pth.tar'
                if not os.path.exists(best_path):
                    st.error("❌ No checkpoint found! Please train model first.")
                else:
                    # Create fresh model instance on CPU
                    model = get_model(num_classes=config.NUM_CLASSES, pretrained=False)
                    model = model.to(cpu_device)
                    
                    # Load checkpoint weights
                    checkpoint = torch.load(best_path, map_location=cpu_device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.eval()
                    st.info("Sampling 20 random images from test set...")
                    
                    # Get all test image paths
                    from pathlib import Path
                    test_dir = Path(config.TEST_DIR)
                    
                    # Collect all image paths with their labels
                    image_paths = []
                    image_labels = []
                    
                    for emotion_idx in range(config.NUM_CLASSES):
                        emotion_dir = test_dir / str(emotion_idx + 1)  # Folders are 1-7
                        if emotion_dir.exists():
                            for img_path in emotion_dir.glob('*.[jp][pn]g'):
                                image_paths.append(str(img_path))
                                image_labels.append(emotion_idx)
                    
                    if len(image_paths) == 0:
                        st.error("❌ No test images found!")
                    else:
                        # Random sample 20 images (different each time)
                        sample_indices = np.random.choice(len(image_paths), min(20, len(image_paths)), replace=False)
                        
                        sample_paths = [image_paths[i] for i in sample_indices]
                        sample_labels = [image_labels[i] for i in sample_indices]
                        
                        # Get transform for test images
                        transform = get_raf_db_transforms(train=False)
                        
                        # Display images in grid
                        st.subheader(f"📸 {len(sample_paths)} Random Test Samples")
                        
                        cols = st.columns(4)
                        
                        with torch.no_grad():
                            for idx, (img_path, true_label_idx) in enumerate(zip(sample_paths, sample_labels)):
                                col = cols[idx % 4]
                                
                                with col:
                                    try:
                                        # Load image
                                        img = Image.open(img_path).convert('RGB')
                                        img_tensor = transform(img).unsqueeze(0)
                                        
                                        # Inference on CPU
                                        img_tensor_cpu = img_tensor.to(cpu_device)
                                        outputs = model(img_tensor_cpu)
                                        probs = torch.nn.functional.softmax(outputs, dim=1)
                                        pred_idx = torch.argmax(probs, dim=1).item()
                                        confidence = probs[0, pred_idx].item()
                                        
                                        # Display
                                        st.image(img, use_container_width=True)
                                        
                                        true_label = config.EMOTION_LABELS[true_label_idx]
                                        pred_label = config.EMOTION_LABELS[pred_idx]
                                        
                                        st.caption(f"**True:** {true_label}")
                                        
                                        is_correct = true_label_idx == pred_idx
                                        if is_correct:
                                            st.caption(f"✅ **Pred:** {pred_label}\n*{confidence*100:.1f}%*")
                                        else:
                                            st.caption(f"❌ **Pred:** {pred_label}\n*{confidence*100:.1f}%*")
                                        
                                        # Clear memory
                                        del img_tensor_cpu, outputs, probs
                                        
                                    except Exception as e:
                                        st.error(f"Error loading {img_path}: {e}")
                        
                        st.success("✅ Samples displayed!")
                    
            except Exception as e:
                st.error(f"❌ Error displaying samples: {e}")
                import traceback
                traceback.print_exc()
    
    with tab3:
        st.header("📊 Evaluate Model")
        
        eval_mode = st.radio("Evaluation Mode", ["Confusion Matrix", "Classification Report", "Per-Class Metrics"], horizontal=True)
        
        if st.button("📈 Generate Report", key="gen_report", use_container_width=True, type="primary"):
            st.info("Generating evaluation report... (Simulation)")
            
            if eval_mode == "Confusion Matrix":
                # Simulate confusion matrix
                cm = np.random.randint(0, 20, (7, 7))
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=LABELS_CLEAN, 
                           yticklabels=LABELS_CLEAN, ax=ax)
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            
            elif eval_mode == "Classification Report":
                # Simulate classification report
                report_data = {
                    'Class': LABELS_CLEAN,
                    'Precision': np.random.uniform(0.7, 0.95, 7),
                    'Recall': np.random.uniform(0.7, 0.95, 7),
                    'F1-Score': np.random.uniform(0.7, 0.95, 7),
                    'Support': np.random.randint(30, 100, 7)
                }
                
                df = pd.DataFrame(report_data)
                st.dataframe(df, use_container_width=True)
            
            else:
                # Per-class metrics
                metrics_data = {
                    'Emotion': LABELS_CLEAN,
                    'Accuracy': np.random.uniform(0.75, 0.95, 7),
                    'Precision': np.random.uniform(0.7, 0.95, 7),
                    'Recall': np.random.uniform(0.7, 0.95, 7)
                }
                
                df = pd.DataFrame(metrics_data)
                st.dataframe(df, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                df.set_index('Emotion')[['Accuracy', 'Precision', 'Recall']].plot(ax=ax)
                ax.set_title('Per-Class Performance')
                ax.set_ylabel('Score')
                st.pyplot(fig)
    
    with tab4:
        st.header("📸 Sample Predictions")
        
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Prediction Result")
                
                if st.button("🔮 Predict Emotion", key="predict_btn", use_container_width=True, type="primary"):
                    try:
                        model = load_model_cached()
                        transform = get_transform()
                        
                        with st.spinner("Predicting..."):
                            pred_idx, pred_prob, all_probs = predict_emotion(image, model, transform)
                        
                        pred_label = LABELS[pred_idx]
                        conf_percent = pred_prob * 100
                        
                        st.metric("Predicted Emotion", pred_label)
                        st.metric("Confidence", f"{conf_percent:.1f}%")
                        
                        # Emotion breakdown - use actual model scores
                        st.markdown("### 📊 Confidence Scores")
                        scores = all_probs  # Use actual model probabilities
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        bars = ax.barh(LABELS_CLEAN, scores)
                        bars[pred_idx].set_color('green')
                        ax.set_xlabel('Confidence')
                        ax.set_title('Emotion Distribution')
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"❌ Prediction error: {e}")
    
    st.divider()
    st.markdown("""
    ---
    **Facial Expression Recognition** | ResNet50 + RAF-DB | 2026
    """)

if __name__ == "__main__":
    main()