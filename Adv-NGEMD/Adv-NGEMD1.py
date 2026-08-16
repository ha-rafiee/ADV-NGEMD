import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import pickle

# Import the YeNet Model class
from models.YeNet import Model  # Ensure this path is correct
import config as c  # Import the config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    model = Model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Stego image not found at: {image_path}")

    if c.stego_img_channel == 1:
        image = Image.open(image_path).convert('L')  # Grayscale
    elif c.stego_img_channel == 3:
        image = Image.open(image_path).convert('RGB')  # RGB
    else:
        raise ValueError(f"Unsupported number of channels: {c.stego_img_channel}")

    transform = transforms.Compose([
        transforms.Resize((c.stego_img_height, c.stego_img_height)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def sparse_pgd_attack(model, image, target_label, eps, alpha, iters, k):
    """
    Performs a targeted PGD attack with sparsity constraint.
    Only updates the top-k pixels with the highest gradient magnitude.
    """
    model.eval()
    image = image.clone().detach().to(device)
    original_image = image.clone().detach()
    image.requires_grad = True

    for i in range(iters):
        outputs = model(image)
        loss = nn.CrossEntropyLoss()(outputs, target_label)
        model.zero_grad()
        loss.backward()

        # Check if gradients are computed
        if image.grad is None:
            print(f"Iteration {i+1}: Gradient is None!")
            break

        grad = image.grad.data

        # Compute the gradient magnitude per pixel (sum over channels)
        grad_abs = grad.abs().sum(dim=1)  # Shape: [1, H, W]
        grad_abs = grad_abs.view(grad_abs.size(0), -1)  # Shape: [1, H*W]

        # Get the top-k indices
        topk_values, topk_indices = torch.topk(grad_abs, k, dim=1)

        # Create a mask for the top-k indices
        mask = torch.zeros_like(grad_abs)
        mask.scatter_(1, topk_indices, 1)

        # Reshape mask to match grad shape
        mask = mask.view(grad.size(0), 1, grad.size(2), grad.size(3))  # Shape: [1, 1, H, W]

        # Apply mask to the gradient (broadcasting over channels)
        masked_grad = grad * mask  # Shape: [1, C, H, W]

        # Update the image
        image = image.detach() + alpha * masked_grad.sign()
        image = torch.min(torch.max(image, original_image - eps), original_image + eps)
        image = torch.clamp(image, 0.0, 1.0).detach()
        image.requires_grad = True

        # Check if attack is successful
        with torch.no_grad():
            output = model(image)
            pred_label = output.argmax(dim=1)
            if pred_label.item() == target_label.item():
                print(f"Attack succeeded at iteration {i+1}")
                break

    return image

def extract_attack_pattern(original_image, perturbed_image):
    # Detach the tensor before converting to NumPy
    delta = (perturbed_image - original_image).detach().squeeze(0).cpu().numpy()
    if c.stego_img_channel == 1:
        delta = delta  # No need to process further, it's already grayscale
    else:
        delta = np.transpose(delta, (1, 2, 0))
    changed_indices = np.where(np.abs(delta) > 0)
    changes = delta[changed_indices]
    attack_pattern = list(zip(*changed_indices, changes))
    return attack_pattern


def main():
    # Parameters
    stego_image_path = '/content/drive/MyDrive/Steganalysis_Project/Paper3_Test/stego/7.pgm'  # Replace with your stego image path
    checkpoint_path = '/content/drive/MyDrive/Steganalysis_Project/checkpoint_YeNet/checkpoint_100.pt'  # Replace with your model checkpoint path
    #EPSILON = 0.03     ALPHA = 0.005      NUM_ITER = 100     K = 1000
    epsilon = 0.03      # Maximum perturbation
    alpha = 0.005       # Attack step size
    num_iter = 100      # Number of attack iterations
    k = 1000            # Maximum number of pixels to change
    output_dir = '/content/drive/MyDrive/Steganalysis_Project/Paper3_Test/Yenet_adv_pgd'

    # Load the model
    model = load_model(checkpoint_path)

    # Load the stego image
    image = load_image(stego_image_path)

    # Set target label to 'cover' (label 0)
    target_label = torch.tensor([0], dtype=torch.long).to(device)

    # Initial prediction
    with torch.no_grad():
        output = model(image)
        probs = nn.functional.softmax(output, dim=1)
        pred_label = output.argmax(dim=1)
        print(f"Initial prediction - Cover: {probs[0, 0]:.4f}, Stego: {probs[0, 1]:.4f}")
        if pred_label.item() == target_label.item():
            print("The image is already classified as cover.")
            return

    # Perform the attack
    print("\nStarting adversarial attack...")
    perturbed_image = sparse_pgd_attack(
        model=model,
        image=image,
        target_label=target_label,
        eps=epsilon,
        alpha=alpha,
        iters=num_iter,
        k=k
    )

    # Final prediction
    with torch.no_grad():
        output = model(perturbed_image)
        probs = nn.functional.softmax(output, dim=1)
        pred_label = output.argmax(dim=1)
        print(f"\nFinal prediction - Cover: {probs[0, 0]:.4f}, Stego: {probs[0, 1]:.4f}")
        if pred_label.item() == target_label.item():
            print("Attack successful: The adversarial image is classified as cover.")
        else:
            print("Attack failed: The adversarial image is still classified as stego.")

    # Extract attack pattern
    attack_pattern = extract_attack_pattern(image, perturbed_image)
    total_changes = len(attack_pattern)
    print(f"\nTotal Pixel Changes: {total_changes}")
    print(f"Attack Pattern (First 10 changes): {attack_pattern[:10]}")
    print(f"stego_img_channel: {c.stego_img_channel}")
    # Save the adversarial image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Detach the tensor before converting to NumPy
    perturbed_image_np = perturbed_image.detach().squeeze(0).cpu().numpy()
    if c.stego_img_channel == 1:
        # Remove the channel dimension
        perturbed_image_np = perturbed_image_np[0]  # Shape: (H, W)
        perturbed_image_np = (perturbed_image_np * 255).astype(np.uint8)
        perturbed_image_pil = Image.fromarray(perturbed_image_np, mode='L')
        print('ok')
    elif c.stego_img_channel == 3:
        # Transpose the axes to (H, W, C)
        perturbed_image_np = np.transpose(perturbed_image_np, (1, 2, 0))
        perturbed_image_np = (perturbed_image_np * 255).astype(np.uint8)
        perturbed_image_pil = Image.fromarray(perturbed_image_np, mode='RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {c.stego_img_channel}")

    adv_image_path = os.path.join(output_dir, 'adv_' + os.path.basename(stego_image_path))
    perturbed_image_pil.save(adv_image_path)
    print(f"Saved adversarial image at: {adv_image_path}")

    # Save attack pattern
    attack_pattern_path = os.path.join(output_dir, 'attack_pattern.pkl')
    with open(attack_pattern_path, 'wb') as f:
        pickle.dump(attack_pattern, f)
    print(f"Saved attack pattern at: {attack_pattern_path}")

if __name__ == '__main__':
    main()