import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from models.YeNet import Model  # Assuming your model implementation is in models/SRNet.py

def load_image(image_path):
    """Load and preprocess a PGM image."""
    # Open the image and convert it to grayscale
    img = Image.open(image_path).convert('L')
    #img = img.resize((512, 512), Image.ANTIALIAS)  # Resize to 512x512 if needed

    # Convert to a NumPy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Convert to a PyTorch tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 512, 512)

    # Repeat the single channel to create three channels
    img_tensor = img_tensor.repeat(1, 3, 1, 1)  # Shape: (1, 3, 512, 512)

    return img_tensor

def calculate_gradient(model, image_tensor):
    """Calculate the gradient of the 'stego' score with respect to the input image."""
    # Enable gradients for the input tensor
    image_tensor.requires_grad = True

    # Forward pass through the model
    output = model(image_tensor)

    # Extract the "stego" score (assume index 1 corresponds to "stego")
    stego_score = output[0, 1]

    # Backward pass to compute gradients
    stego_score.backward()

    # Get the gradient of the input image
    gradient = image_tensor.grad[0, 0].cpu().detach().numpy()  # Extract gradient of the first channel

    return gradient
def display_gradient_matrix(gradient):
    """Display the gradient as numerical values in a matrix format."""
    print("Gradient Matrix (512x512):")
    np.set_printoptions(precision=4, suppress=True)  # Format output for readability
    for row in gradient:
        print(" ".join(f"{val:.4f}" for val in row))  # Print each row with 4 decimal places

def display_gradient(gradient):
    """Display the gradient as a 512x512 matrix."""
    plt.figure(figsize=(8, 8))
    plt.imshow(gradient, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title("Gradient Matrix (512x512)")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()
def analyze_gradients(gradient):
    non_zero_gradients = gradient[np.abs(gradient) > 0]
    abs_gradient = np.abs(gradient)
    mean_abs_non_zero = np.mean(np.abs(non_zero_gradients))
    mean_abs_greater_0001 = np.mean(abs_gradient[abs_gradient > 0.0001])
    mean_abs_greater_1 = np.mean(abs_gradient[abs_gradient > 1])
    greater_1_count = np.sum(gradient > 1)
    less_neg_1_count = np.sum(gradient < -1)
    greater_0001_count = np.sum(gradient > 0.0001)
    less_neg_0001_count = np.sum(gradient < -0.0001)
    stats = {
        "min_non_zero": np.min(non_zero_gradients),
        "max_non_zero": np.max(non_zero_gradients),
        "mean_abs_non_zero": mean_abs_non_zero,
        "mean_abs_greater_0001": mean_abs_greater_0001,
        "mean_abs_greater_1": mean_abs_greater_1,
        "greater_0001_count": greater_0001_count,
        "less_neg_0001_count": less_neg_0001_count,
        "greater_1_count": greater_1_count,
        "less_neg_1_count": less_neg_1_count,
    }
    return stats

if __name__ == "__main__":
    # Path to the PGM image and model checkpoint
    cover_path = "/content/drive/MyDrive/Steganalysis_Project/testJSMA/cover/7000.pgm"
    stego_path = "/content/drive/MyDrive/Steganalysis_Project/testJSMA/stego/7000.pgm"
    checkpoint_path = "/content/drive/MyDrive/Steganalysis_Project/checkpoint_YeNet/checkpoint_100.pt"

    # Load the model
    model = Model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # Load and preprocess the image
    cover_tensor = load_image(cover_path)
    stego_tensor = load_image(stego_path)

    # Calculate the gradient
    gradient_cover = calculate_gradient(model, cover_tensor)
    gradient_stego = calculate_gradient(model, stego_tensor)
    gradient_diff = gradient_cover - gradient_stego
    #print(gradient_diff)
    # Display the gradient as a matrix
    #display_gradient(gradient_diff)
    #display_gradient_matrix(gradient_diff)
    stats = analyze_gradients(gradient_diff)
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")
