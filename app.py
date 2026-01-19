import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from gradcam import GradCAM
import matplotlib.pyplot as plt
import cv2
from torchvision import models
import torch.nn as nn

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("pneumonia_resnet18_3class.pt", map_location="cpu"))
model.eval()

target_layer = model.layer4
gradcam = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred_class = torch.argmax(output, dim=1).item()

    label_map = {
        0: "Normal",
        1: "Pneumonia (Bacterial)",
        2: "Pneumonia (Viral)"
    }
    label = label_map[pred_class]

    # GradCAM
    cam = gradcam.generate(img_tensor)
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = cv2.cvtColor(
        np.array(image.resize((224, 224))),
        cv2.COLOR_RGB2BGR
            )

    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


    return label, Image.fromarray(overlay)

gr.Interface(
    fn=predict,
    inputs="image",
    outputs=["text", "image"],
    title="Pneumonia Detector with GradCAM"
).launch()
