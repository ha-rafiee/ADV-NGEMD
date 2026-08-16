# YeNet Adversarial EMD Embedding (Group of 3 Pixels) 0.935bpp
import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# Import your YeNet model
from models.YeNet import Model  # Adjust as needed for your file structure

############################################
# Part 1: Gradient Calculation & Classification
############################################

def load_image_for_model(image_path):
    """
    Loads an image in grayscale, normalizes to [0,1], then repeats channels
    to create a shape (1,3,H,W) for YeNet. We do NOT embed in channels 1 and 2,
    but we still need them for YeNet's input format.
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Torch shape: (batch=1, channel=1, H, W)
    tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    # YeNet expects 3 channels, so repeat
    tensor = tensor.repeat(1, 3, 1, 1)

    return img, tensor

def calculate_gradient(model, image_tensor):
    """
    Calculate the gradient of the 'stego' score (index=1) w.r.t. the *first channel*.
    We'll only use channel 0's gradient for embedding in the first channel.
    """
    image_tensor.requires_grad = True
    output = model(image_tensor)
    stego_score = output[0, 1]  # 'stego' class logit
    stego_score.backward()

    # Extract gradient for the FIRST channel only (index=0)
    grad = image_tensor.grad[0, 0].cpu().detach().numpy()
    return grad

def classify_image_direct(model, stego_array):
    """
    Classify the final stego image directly, with no further preprocessing.
    - stego_array: shape (H, W, 3) or (H, W) for grayscale, but we'll ensure 3-ch is used if needed.
    """
    if stego_array.ndim == 2:
        # grayscale => shape (H, W)
        # expand to (H, W, 3)
        H, W = stego_array.shape
        rgb_array = np.stack([stego_array, stego_array, stego_array], axis=-1)
    else:
        # assume (H, W, 3)
        rgb_array = stego_array
        H, W, _ = rgb_array.shape

    # Normalize to [0,1]
    tensor_in = torch.tensor(rgb_array, dtype=torch.float32) / 255.0
    # Reorder to (1, 3, H, W)
    tensor_in = tensor_in.permute(2, 0, 1).unsqueeze(0)  # shape: (1,3,H,W)

    with torch.no_grad():
        output = model(tensor_in)
        pred = output.argmax(dim=1).item()  # 0 (cover) or 1 (stego)
    return pred

############################################
# Part 2: EMD + Adversarial Embedding (Only 1st Channel, Group of 3 Pixels)
############################################

def Extraction(v_img, C, n, g):
    """
    Extraction function for a group of n pixels.
      v_img: list/array of pixel values of length n.
      C: coefficient list of length n.
      g: modulus.
    """
    temp = 0
    for i in range(n):
        temp += v_img[i] * C[i]
    return int(temp) % g

def sign_matches(dx, grad_s):
    """
    Check whether a proposed change (dx) is in the OPPOSITE direction of the gradient.
      - If grad_s is positive, we require dx < 0.
      - If grad_s is negative, we require dx > 0.
      - If grad_s is near zero, any dx is accepted.
    """
    if grad_s == '+':
        return dx < 0
    elif grad_s == '-':
        return dx > 0
    else:
        return True

def get_grad_sign(g_val, threshold=1e-6):
    """
    Returns '+' if g_val > threshold, '-' if g_val < -threshold, else '0'.
    """
    if g_val > threshold:
        return '+'
    elif g_val < -threshold:
        return '-'
    else:
        return '0'

def lookupT231(d):

    lookup_table = {
        6: [-1, 0, 0],
        5: [0, -1, 0],
        4: [0, 0, -1],
        0: [0, 0, 0],
        3: [0, 0, 1],
        2: [0, 1, 0],
        1: [1, 0, 0]
    }
    return lookup_table.get(d, [0, 0, 0])

def find_alternative_changes_n(x, s, grad_sign, C, mod, max_range=8):
    """
    Heuristic search for alternative changes for a group of n pixels.
    x: list/array of current pixel values (length n)
    s: desired extracted secret value (an integer in 0...mod-1)
    grad_sign: list of gradient signs for each pixel (length n)
    C: list of coefficients (length n)
    mod: modulus
    max_range: maximum absolute change to try per pixel.

    The function searches (via recursive backtracking) for a vector dx (length n)
    such that:
      - (sum_i C[i]*(x[i] + dx[i])) mod mod == s, and
      - For each pixel i, dx[i] is in the allowed range and is in the opposite direction
        of the gradient (as determined by sign_matches).
    Returns the vector dx (as a list) if found, or None if no solution is found within the range.
    """
    n = len(x)
    # Compute the current extraction sum
    d_orig = sum([C[i]*x[i] for i in range(n)]) % mod
    target = (s - d_orig) % mod
    best_solution = None
    best_cost = float('inf')

    # Try increasing candidate ranges from 1 to max_range.
    for r in range(1, max_range+1):
        dx_range = range(-r, r+1)
        best_solution = None
        best_cost = float('inf')
        def rec(i, current_sum, current_solution, current_cost):
            nonlocal best_solution, best_cost
            if i == n:
                if current_sum % mod == target:
                    if current_cost < best_cost:
                        best_solution = current_solution.copy()
                return
            for dx in dx_range:
                if not sign_matches(dx, grad_sign[i]):
                    continue
                new_cost = current_cost + abs(dx)
                if new_cost >= best_cost:
                    continue
                rec(i+1, current_sum + C[i]*dx, current_solution + [dx], new_cost)
        rec(0, 0, [], 0)
        if best_solution is not None:
            return best_solution
    return None

def embed_adversarial_emd(cover_array, gradient, m=7, C=[1, 2, 3]):
    """
    Embed using EMD with adversarial (gradient-based) adjustments, only in the first channel.
    Now processing groups of 3 pixels.

    Parameters:
      cover_array: 2D NumPy array (grayscale image).
      gradient: gradient array (same shape as cover_array).
      m: modulus (set to 7 for 3-pixel groups).
      C: coefficient list (length 3).

    Returns the final stego image (2D, shape (H,W)) and prints statistics.
    """
    H, W = cover_array.shape
    l = H * W
    n = 3  # group size changed to 3
    # Flatten the image and gradient arrays.
    img_flat = cover_array.flatten().astype(np.float64)
    grad_flat = gradient.flatten()

    ls = l // n  # number of complete groups
    # Generate a random secret message (values 0..m-1) for each group.
    secret = np.random.randint(0, m, ls)

    count_already = 0
    count_align = 0
    count_heuristic = 0
    count_fallback = 0

    idx = 0
    for j in range(ls):
        group = img_flat[idx: idx+n]
        s_val = secret[j]

        # Check if current group already extracts to the secret.
        curr_ext = Extraction(group, C, n, m)
        if curr_ext == s_val:
            count_already += 1
            idx += n
            continue

        # Compute default table change.
        sum_orig = sum([C[i]*group[i] for i in range(n)]) % m
        needed_diff = (s_val - sum_orig) % m
        default_chg = lookupT231(needed_diff)

        # Get gradient signs for the n pixels.
        grad_signs = [get_grad_sign(grad_flat[idx+i]) for i in range(n)]

        # Check if the default change is adversarial for every pixel.
        if all(sign_matches(default_chg[i], grad_signs[i]) for i in range(n)):
            new_group = group + np.array(default_chg)
            if np.all(new_group >= 0) and np.all(new_group <= 255):
                img_flat[idx: idx+n] = new_group
                count_align += 1
                idx += n
                continue

        # Try to find alternative changes that are adversarial.
        alt_chg = find_alternative_changes_n(group, s_val, grad_signs, C, m)
        if alt_chg is None:
            # Fallback: use the default table change even though it is not adversarial.
            new_group = group + np.array(default_chg)
            if np.all(new_group >= 0) and np.all(new_group <= 255):
                img_flat[idx: idx+n] = new_group
            count_fallback += 1
        else:
            # Use the heuristic alternative change.
            new_group = group + np.array(alt_chg)
            if np.all(new_group >= 0) and np.all(new_group <= 255):
                img_flat[idx: idx+n] = new_group
            count_heuristic += 1

        idx += n

    stego_array = img_flat.reshape(H, W).astype(np.uint8)

    # Print statistics.
    print(f"Total pixel groups: {ls}")
    print(f"Groups already correct (no change needed): {count_already}")
    print(f"Groups used default changes (adversarial alignment): {count_align}")
    print(f"Groups used heuristic changes: {count_heuristic}")
    print(f"Groups used fallback default: {count_fallback}")

    return stego_array

############################################
# Part 3: Main Script (Processing a Folder of Images)
############################################

def main():
    # Define paths.
    cover_dir = "/content/drive/MyDrive/Steganalysis_Project/Paper3_Test/cover"
    checkpoint_path = "/content/drive/MyDrive/Steganalysis_Project/checkpoint_YeNet/checkpoint_100.pt"

    # Create an output directory for stego images.
    stego_out_dir = "/content/drive/MyDrive/Steganalysis_Project/Paper3_Test/YeNet_stego_back3"
    os.makedirs(stego_out_dir, exist_ok=True)

    # 1) Load YeNet once.
    model = Model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # 2) List all image files in the cover directory.
    image_files = [f for f in os.listdir(cover_dir) if f.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg', '.bmp'))]

    # Process each image.
    for file_name in image_files:
        image_path = os.path.join(cover_dir, file_name)
        print(f"\nProcessing image: {image_path}")

        # Load cover image & compute gradient for channel 0.
        cover_img, cover_tensor = load_image_for_model(image_path)
        gradient_cover = calculate_gradient(model, cover_tensor)

        # 'cover_img' is a 2D grayscale. Convert it to a NumPy array.
        cover_array = np.array(cover_img, dtype=np.uint8)

        # Perform EMD embedding (in 1st channel) aligned with the adversarial objective.
        # Using m=7 and C of length 3 for 3-pixel groups.
        stego_array_1ch = embed_adversarial_emd(cover_array, gradient_cover, m=7, C=[1, 2, 3])

        # Create a final 3-channel image for YeNet.
        H, W = stego_array_1ch.shape
        stego_3ch = np.stack([stego_array_1ch, stego_array_1ch, stego_array_1ch], axis=-1)

        # Classify the final stego image with no extra preprocessing.
        pred_label = classify_image_direct(model, stego_3ch)
        if pred_label == 0:
            print("YeNet classifies the final image as COVER.")
        else:
            print("YeNet classifies the final image as STEGO.")

        # Save the final stego image.
        out_name = f"advEMD3_{file_name}"
        out_path = os.path.join(stego_out_dir, out_name)
        Image.fromarray(stego_array_1ch).save(out_path)
        print(f"Stego image saved to: {out_path}")
        psnr_value = peak_signal_noise_ratio(cover_array, stego_array_1ch, data_range=255)
        ssim_value = structural_similarity(cover_array, stego_array_1ch, data_range=255)
        print(f"PSNR (Cover vs Stego): {psnr_value}")
        print(f"SSIM (Cover vs Stego): {ssim_value}")


if __name__ == "__main__":
    main()
