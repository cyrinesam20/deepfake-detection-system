"""
GradCAM++ implementation pour main.py
Remplace complètement la classe GradCAM existante
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
import io
from PIL import Image


class GradCAMPlusPlus:
    """
    GradCAM++ — Plus précis que Grad-CAM standard
    Meilleure localisation des zones manipulées
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backprop
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # GradCAM++ weights — plus précis que GradCAM standard
        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Calcul des poids GradCAM++
        grad_sq = gradients ** 2
        grad_cube = gradients ** 3
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)

        # Dénominateur
        denominator = 2 * grad_sq + sum_activations * grad_cube
        denominator = torch.where(
            denominator != 0,
            denominator,
            torch.ones_like(denominator)
        )

        # Coefficients alpha
        alpha = grad_sq / denominator

        # Weights
        weights = (alpha * torch.relu(gradients)).sum(dim=(1, 2))

        # Heatmap
        heatmap = torch.zeros(activations.shape[1:])
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = torch.relu(heatmap).numpy()

        # Normalisation robuste
        if heatmap.max() > 0:
            p_low = np.percentile(heatmap, 20)
            p_high = np.percentile(heatmap, 99)
            heatmap = np.clip(heatmap, p_low, p_high)
            if p_high > p_low:
                heatmap = (heatmap - p_low) / (p_high - p_low)
            else:
                heatmap = heatmap / heatmap.max()

        probs = torch.softmax(output, dim=1)[0]
        confidence = probs[class_idx].item()

        return heatmap, class_idx, confidence, probs.detach().numpy()


def generate_overlay_gradcampp(original_img, heatmap, alpha=0.45):
    """Overlay GradCAM++ haute qualité"""
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # Améliorer le contraste
    heatmap_resized = np.power(heatmap_resized, 0.75)

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img_array = np.array(original_img.resize((224, 224)))
    overlay = np.clip(
        (1 - alpha) * img_array + alpha * heatmap_colored, 0, 255
    ).astype(np.uint8)

    overlay_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_heatmap_b64(heatmap):
    """Heatmap seule en base64"""
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_resized = np.power(heatmap_resized, 0.75)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_img = Image.fromarray(heatmap_colored)
    buffer = io.BytesIO()
    heatmap_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')