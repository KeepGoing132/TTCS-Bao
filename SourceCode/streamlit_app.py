import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import config
from dataset import get_dataloaders

# ============================================================================
# PAGE CONFIG & THEME
# ============================================================================
st.set_page_config(
    page_title="😊 Facial Expression Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://streamlit.io',
        'Report a bug': None,
        'About': "Facial Expression Recognition - ResNet18"
    }
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --success-color: #45B7D1;
        --warning-color: #FFA502;
    }
    
    .main {
        padding-top: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    
    .title-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .title-section h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .title-section p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Hide Deploy and all toolbar buttons including menu */
    [data-testid="stToolbarActionButton"] {
        display: none !important;
    }
    
    header[data-testid="stHeader"] > div:nth-child(2) {
        display: none !important;
    }
    
    /* Hide menu button */
    button[kind="secondary"] {
        display: none !important;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "is_training" not in st.session_state:
    st.session_state.is_training = False
if "stop_training" not in st.session_state:
    st.session_state.stop_training = False

# ============================================================================
# CONSTANTS
# ============================================================================
LABELS = ["😠 Angry", "😒 Disgust", "😨 Fear", "😊 Happy", "😐 Neutral", "😢 Sad", "😲 Surprise"]
LABELS_CLEAN = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MODEL_NAME = "resnet18"
DEVICE = torch.device("cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

# ============================================================================
# MODEL WRAPPER FOR DROPOUT
# ============================================================================
class ResNet18WithDropout(nn.Module):
    """ResNet18 with optional dropout layer"""
    def __init__(self, base_model, dropout_rate=0.0):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x):
        # Forward through ResNet18
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout before final FC layer
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.base.fc(x)
        return x
    
    def eval(self):
        super().eval()
        if self.dropout is not None:
            self.dropout.eval()
        return self
    
    def train(self, mode=True):
        super().train(mode)
        if self.dropout is not None:
            self.dropout.train(mode)
        return self

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
@st.cache_resource
def get_model():
    """Load ResNet18 model"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    return model.to(DEVICE)


def get_transform():
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        return model
    except Exception as e:
        st.error(f"❌ Lỗi tải model: {e}")
        return None


def predict_image(image, model):
    """Predict emotion for a single image"""
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_prob = probs[pred_idx].item()
    
    return pred_idx, pred_prob, probs.cpu().numpy()


def plot_predictions(probs):
    """Create beautiful prediction chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA502', '#95E1D3', '#F38181', '#AA96DA']
    bars = ax.barh(LABELS, probs, color=colors)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob + 0.01, i, f'{prob*100:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_title('Emotion Prediction Confidence', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    plt.tight_layout()
    
    return fig


def plot_confusion_matrix(cm, labels):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', 
                xticklabels=[l.split()[-1] for l in labels],
                yticklabels=[l.split()[-1] for l in labels],
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    return fig


def get_sample_test_images(num_samples=20):
    """Get random sample of test images from all emotions"""
    test_dir = config.TEST_DIR
    all_images = []
    
    for emotion in LABELS_CLEAN:
        emotion_dir = os.path.join(test_dir, emotion)
        if os.path.exists(emotion_dir):
            images = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            for img_file in images:
                all_images.append((emotion, img_file))
    
    # Shuffle and get random samples
    import random
    random.shuffle(all_images)
    return all_images[:num_samples]


# ============================================================================
# MAIN HEADER
# ============================================================================
st.markdown("""
<div class="title-section">
    <h1>😊 Facial Expression Recognition</h1>
    <p>ResNet18 Deep Learning Model • Real-time Emotion Detection</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check for saved models
    best_model_path = "saved_models/resnet18_best.pth"
    latest_model_path = "saved_models/resnet18_latest.pth"
    
    st.divider()
    
    # Training parameters
    st.subheader("📚 Training Parameters")
    epochs = st.slider("Number of Epochs", 1, 100, config.EPOCHS, help="How many epochs to train")
    lr = st.number_input("Learning Rate", 1e-5, 1e-2, config.LR, format="%.2e")
    batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=config.BATCH_SIZE)
    
    st.divider()
    
    # Regularization options
    st.subheader("🔧 Regularization Options")
    
    # Weight-based regularization
    reg_type = st.selectbox(
        "Weight Regularization",
        ["L2", "L1", "None"],
        index=0,
        help="L2 or L1 penalty on weights"
    )
    
    if reg_type in ["L1", "L2"]:
        reg_factor = st.slider(
            f"{reg_type} Factor",
            1e-5, 1e-2, 5e-4,
            format="%.2e",
            help=f"Regularization strength"
        )
    else:
        reg_factor = 0.0
    
    # Dropout option (can combine with L1/L2)
    use_dropout = st.checkbox("✓ Add Dropout Layer", value=False, help="Combine with weight regularization")
    dropout_rate = 0.0
    if use_dropout:
        dropout_rate = st.slider(
            "Dropout Rate",
            0.0, 0.5, 0.3,
            step=0.05,
            help="Probability of dropping neurons"
        )
    
    st.divider()
    
    # Model info
    st.subheader("ℹ️ Model Information & Status")
    
    # Get model status
    model_status = "❌ No model found"
    if os.path.exists(best_model_path):
        model_status = f"✅ Best Model Found\n(saved_models/resnet18_best.pth)"
    elif os.path.exists(latest_model_path):
        model_status = f"✅ Latest Model Found\n(saved_models/resnet18_latest.pth)"
    
    # Build regularization display
    reg_display = f"{reg_type} ({reg_factor:.2e})" if reg_type != "None" else "None"
    if use_dropout:
        reg_display += f" + Dropout ({dropout_rate:.2f})"
    
    # Show comprehensive status
    st.markdown(f"""
    **Model Status:** {model_status}
    
    **Configuration:**
    - Architecture: ResNet-18
    - Classes: {config.NUM_CLASSES}
    - Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}
    - Device: {DEVICE.type.upper()}
    
    **Current Regularization:** {reg_display}
    """)

# ============================================================================
# MAIN TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🎓 Train", "🧪 Test", "📊 Evaluate", "📸 Sample Test Images"])

# ============================================================================
# TAB 1: TRAINING
# ============================================================================
with tab1:
    st.header("🎓 Train Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📖 Hướng dẫn:
        1. Nhấn **▶️ Start Training** để bắt đầu training ResNet18
        2. Theo dõi tiến độ với các metrics real-time
        3. Model tốt nhất sẽ được lưu tự động
        4. Sử dụng model được train trong tab Test & Evaluate
        """)
    
    with col2:
        st.metric("Current Epochs", epochs)
        st.metric("Learning Rate", f"{lr:.2e}")
        st.metric("Batch Size", batch_size)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_from_scratch = st.checkbox("🔄 Train from Scratch", value=False)
    
    start_button_clicked = False
    stop_button_clicked = False
    
    with col2:
        if not st.session_state.is_training:
            start_button_clicked = st.button("▶️ Start Training", use_container_width=True, type="primary")
    
    with col3:
        if st.session_state.is_training:
            stop_button_clicked = st.button("⏹️ Stop Training", use_container_width=True, type="secondary")
    
    # Handle start button
    if start_button_clicked:
        st.session_state.is_training = True
        st.session_state.stop_training = False  # Reset stop flag
        st.rerun()
    
    # Handle stop button
    if stop_button_clicked:
        st.session_state.is_training = False
        st.session_state.stop_training = True
        st.rerun()
    
    if st.session_state.is_training:
        st.info("⏳ Training in progress... This may take a while.")
        
        try:
            # Load data
            train_loader, val_loader, _ = get_dataloaders(MODEL_NAME)
            model = get_model()
            
            # Apply dropout wrapper if enabled
            if use_dropout:
                model = ResNet18WithDropout(model, dropout_rate)
            
            criterion = nn.CrossEntropyLoss()
            
            # Use selected regularization from sidebar
            weight_decay = 0.0
            if reg_type == "L2":
                weight_decay = reg_factor
            # L1 applied in loop, Dropout applied as wrapper
            
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
            
            # Create placeholders for metrics
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.container()
            
            best_val_acc = 0.0
            
            for epoch in range(1, epochs + 1):
                # Check if user stopped training
                if st.session_state.stop_training:
                    st.session_state.stop_training = False
                    break
                
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for images, labels in train_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # Add L1 regularization if selected (L2 handled via weight_decay)
                    if reg_type == "L1" and reg_factor > 0:
                        l1_norm = sum(torch.sum(torch.abs(p)) for p in model.parameters() if p.requires_grad)
                        loss = loss + reg_factor * l1_norm
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * images.size(0)
                    train_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    train_total += labels.size(0)
                
                train_loss /= train_total
                train_acc = train_correct / train_total
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * images.size(0)
                        val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                        val_total += labels.size(0)
                
                val_loss /= val_total
                val_acc = val_correct / val_total
                scheduler.step(val_loss)
                
                # Update progress
                progress = epoch / epochs
                progress_bar.progress(progress)
                
                status_text.write(f"**Epoch {epoch}/{epochs}** - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    os.makedirs("saved_models", exist_ok=True)
                    torch.save(model.state_dict(), "saved_models/resnet18_best.pth")
                
                # Save latest checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_acc": best_val_acc
                }, "saved_models/resnet18_latest.pth")
            
            st.session_state.is_training = False
            st.session_state.stop_training = False
            st.success(f"✅ Training completed! Best Validation Accuracy: **{best_val_acc*100:.2f}%**")
            
        except Exception as e:
            st.session_state.is_training = False
            st.session_state.stop_training = False
            st.error(f"❌ Error during training: {str(e)}")

# ============================================================================
# TAB 2: TEST SINGLE IMAGE
# ============================================================================
with tab2:
    st.header("🧪 Test Single Image")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Upload an Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
    
    with col2:
        st.markdown("### Or Capture Photo")
        use_camera = st.checkbox("📷 Use Camera", value=False)
    
    camera_photo = None
    if use_camera:
        camera_photo = st.camera_input("Take a picture")
    
    st.divider()
    
    image_to_test = uploaded_file or camera_photo
    
    if image_to_test:
        try:
            image = Image.open(image_to_test).convert("RGB")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Input Image", use_container_width=True)
            
            with col2:
                if st.button("🔮 Predict Emotion", use_container_width=True, type="primary"):
                    with st.spinner("🔄 Analyzing..."):
                        model = get_model()
                        best_model_path = "saved_models/resnet18_best.pth"
                        
                        if not os.path.exists(best_model_path):
                            st.error("❌ No trained model found! Please train the model first.")
                        else:
                            model = load_checkpoint(model, best_model_path)
                            if model:
                                pred_idx, pred_prob, all_probs = predict_image(image, model)
                                
                                # Display prediction
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Predicted Emotion", LABELS[pred_idx])
                                with col2:
                                    st.metric("Confidence", f"{pred_prob*100:.1f}%")
                                with col3:
                                    confidence_level = "🟢 High" if pred_prob > 0.7 else "🟡 Medium" if pred_prob > 0.5 else "🔴 Low"
                                    st.metric("Confidence Level", confidence_level)
                                
                                st.divider()
                                
                                # Plot predictions
                                st.subheader("📊 Confidence Scores")
                                fig = plot_predictions(all_probs)
                                st.pyplot(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

# ============================================================================
# TAB 3: EVALUATE ON TEST SET
# ============================================================================
with tab3:
    st.header("📊 Evaluate on Test Set")
    
    st.markdown("""
    ### 📈 Model Performance Metrics
    This tab evaluates the model's performance on the entire test dataset.
    """)
    
    st.divider()
    
    if st.button("▶️ Run Evaluation", use_container_width=True, type="primary"):
        try:
            with st.spinner("⏳ Evaluating model on test set..."):
                # Load data and model
                _, _, test_loader = get_dataloaders(MODEL_NAME)
                model = get_model()
                best_model_path = "saved_models/resnet18_best.pth"
                
                if not os.path.exists(best_model_path):
                    st.error("❌ No trained model found! Please train the model first.")
                else:
                    model = load_checkpoint(model, best_model_path)
                    if model:
                        criterion = nn.CrossEntropyLoss()
                        
                        # Evaluation loop
                        model.eval()
                        total_loss = 0.0
                        correct = 0
                        total = 0
                        y_true = []
                        y_pred = []
                        
                        with torch.no_grad():
                            for images, labels in test_loader:
                                images, labels = images.to(DEVICE), labels.to(DEVICE)
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                                total_loss += loss.item() * images.size(0)
                                preds = outputs.argmax(dim=1)
                                correct += (preds == labels).sum().item()
                                total += labels.size(0)
                                y_true.extend(labels.cpu().numpy())
                                y_pred.extend(preds.cpu().numpy())
                        
                        # Calculate metrics
                        avg_loss = total_loss / total
                        accuracy = correct / total
                        
                        # Confusion matrix
                        cm = np.zeros((len(LABELS_CLEAN), len(LABELS_CLEAN)))
                        for t, p in zip(y_true, y_pred):
                            cm[t][p] += 1
                        
                        # Per-class metrics
                        precision = []
                        recall = []
                        f1 = []
                        
                        for i in range(len(LABELS_CLEAN)):
                            tp = cm[i][i]
                            fp = cm[:, i].sum() - tp
                            fn = cm[i, :].sum() - tp
                            
                            p = tp / (tp + fp) if (tp + fp) > 0 else 0
                            r = tp / (tp + fn) if (tp + fn) > 0 else 0
                            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                            
                            precision.append(p)
                            recall.append(r)
                            f1.append(f)
                        
                        # Display overall metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📊 Accuracy", f"{accuracy*100:.2f}%")
                        with col2:
                            st.metric("📉 Loss", f"{avg_loss:.4f}")
                        with col3:
                            st.metric("✅ Test Samples", f"{total}")
                        with col4:
                            st.metric("🎯 Correct", f"{correct}")
                        
                        st.divider()
                        
                        # Per-class metrics
                        st.subheader("📈 Per-Class Metrics")
                        metrics_df = {
                            "Emotion": LABELS,
                            "Precision": [f"{p*100:.1f}%" for p in precision],
                            "Recall": [f"{r*100:.1f}%" for r in recall],
                            "F1-Score": [f"{f*100:.1f}%" for f in f1]
                        }
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        st.divider()
                        
                        # Confusion matrix visualization
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("🔥 Confusion Matrix")
                            fig_cm = plot_confusion_matrix(cm, LABELS_CLEAN)
                            st.pyplot(fig_cm, use_container_width=True)
                        
                        with col2:
                            st.subheader("📊 Per-Class Performance")
                            fig_bar, ax = plt.subplots(figsize=(10, 6))
                            x = np.arange(len(LABELS))
                            width = 0.25
                            ax.bar(x - width, precision, width, label='Precision', color='#FF6B6B', alpha=0.8)
                            ax.bar(x, recall, width, label='Recall', color='#4ECDC4', alpha=0.8)
                            ax.bar(x + width, f1, width, label='F1-Score', color='#45B7D1', alpha=0.8)
                            ax.set_ylabel('Score', fontweight='bold')
                            ax.set_title('Per-Class Metrics', fontweight='bold', fontsize=14)
                            ax.set_xticks(x)
                            ax.set_xticklabels([l.split()[-1] for l in LABELS], fontsize=9)
                            ax.legend()
                            ax.set_ylim([0, 1])
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig_bar, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error during evaluation: {str(e)}")

# ============================================================================
# TAB 4: SAMPLE TEST IMAGES
# ============================================================================
with tab4:
    st.header("📸 Predict on 20 Random Test Images")
    
    st.markdown("""
    ### Predict Emotions on Random Test Images
    Load 20 random images from the test set and get emotion predictions for each one.
    """)
    
    st.divider()
    
    if st.button("🔄 Load & Predict on 20 Random Images", use_container_width=True, type="primary"):
        with st.spinner("📷 Loading and predicting..."):
            try:
                # Check if model exists
                best_model_path = "saved_models/resnet18_best.pth"
                if not os.path.exists(best_model_path):
                    st.error("❌ No trained model found! Please train the model first.")
                else:
                    # Load model
                    model = get_model()
                    model = load_checkpoint(model, best_model_path)
                    
                    if model:
                        # Get sample images
                        samples = get_sample_test_images(num_samples=20)
                        
                        if not samples:
                            st.warning("⚠️ No test images found in the test directory")
                        else:
                            st.success(f"✅ Loaded {len(samples)} random test images")
                            st.divider()
                            
                            # Display images in a grid with predictions
                            cols = st.columns(4)  # 4 columns per row
                            
                            correct = 0
                            total = len(samples)
                            
                            for idx, (true_emotion, img_file) in enumerate(samples):
                                with cols[idx % 4]:
                                    img_path = os.path.join(config.TEST_DIR, true_emotion, img_file)
                                    
                                    if os.path.exists(img_path):
                                        try:
                                            # Load and predict
                                            img = Image.open(img_path).convert("RGB")
                                            pred_idx, pred_prob, all_probs = predict_image(img, model)
                                            pred_emotion = LABELS_CLEAN[pred_idx]
                                            
                                            # Check if prediction is correct
                                            is_correct = pred_emotion == true_emotion
                                            if is_correct:
                                                correct += 1
                                            
                                            # Display image
                                            st.image(img, use_container_width=True)
                                            
                                            # Display prediction result - 3 lines format
                                            pred_label = LABELS[pred_idx]
                                            true_label = LABELS[LABELS_CLEAN.index(true_emotion)]
                                            conf_percent = pred_prob*100
                                            
                                            result_text = f"**Pred:** {pred_label}\n**True:** {true_label}\n**Conf:** {conf_percent:.1f}%"
                                            
                                            if is_correct:
                                                st.success(f"✅ {result_text}")
                                            else:
                                                st.error(f"❌ {result_text}")
                                        
                                        except Exception as e:
                                            st.warning(f"Could not process {img_file}")
                            
                            # Summary
                            st.divider()
                            accuracy = correct / total if total > 0 else 0
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("✅ Correct Predictions", correct)
                            with col2:
                                st.metric("❌ Wrong Predictions", total - correct)
                            with col3:
                                st.metric("🎯 Accuracy", f"{accuracy*100:.1f}%")
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🎯 Facial Expression Recognition System • ResNet18 Deep Learning</p>
    <p style='font-size: 0.9rem;'>Built with Streamlit & PyTorch</p>
</div>
""", unsafe_allow_html=True)
