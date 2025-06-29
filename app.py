import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import torch
from cnn_model import PneumoniaCNN
import numpy as np
from gradcam import GradCAM
import matplotlib.pyplot as plt
import cv2


model = PneumoniaCNN()
model.load_state_dict(torch.load("pneumonia_cnn.pt", map_location="cpu"))
model.eval()

target_layer = model.conv2  # Choose the last conv layer
gradcam = GradCAM(model, target_layer)


def predict(image):

    # Convert NumPy array (from Gradio) to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)


    transform = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        prob = output.item()
    label = "Pneumonia" if prob > 0.5 else "Normal"

    # GradCAM heatmap
    cam = gradcam.generate(img_tensor)


    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(image.resize((150,150)))
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    return label, Image.fromarray(overlay)
gr.Interface(
    fn=predict,
    inputs="image",
    outputs=["text", "image"],
    title="Pneumonia Detector with GradCAM"
).launch()
