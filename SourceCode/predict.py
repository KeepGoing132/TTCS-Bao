import argparse
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

import config

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def get_model(name):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
        return model
    raise ValueError(f"Model không hỗ trợ: {name}")


def get_transform(model_name="resnet18"):
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    ])


def load_image(image_path, model_name="baseline"):
    image = Image.open(image_path).convert("RGB")
    image = get_transform(model_name)(image)
    return image.unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict single image with trained model")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn tới ảnh cần dự đoán")
    parser.add_argument("--model", choices=["baseline", "cbam", "resnet18"], default="baseline",
                        help="Chọn model đã train")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Đường dẫn file checkpoint .pth")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    model = get_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    image = load_image(args.image, args.model).to(device)
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()

    print(f"Dự đoán: {LABELS[pred]} (class {pred})")


if __name__ == "__main__":
    main()
