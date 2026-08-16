# YeNet Adversarial EMD Embedding (Group of 3 Pixels) 2.214
import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
from models.YeNet import Model
# Import your YeNet model (adjust the import path as needed)
from models.YeNet import Model

############################################
# Part 1: Gradient Calculation & Classification
############################################

def load_image_for_model(image_path):
    """
    Loads an image in grayscale, normalizes to [0,1], then repeats channels
    to create a shape (1,3,H,W) for YeNet. (Embedding is performed only in channel 0.)
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Create tensor of shape (1,1,H,W) then repeat to have 3 channels.
    tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    tensor = tensor.repeat(1, 3, 1, 1)
    return img, tensor

def calculate_gradient(model, image_tensor):
    """
    Calculate the gradient of the 'stego' score (index=1) with respect to the first channel.
    """
    image_tensor.requires_grad = True
    output = model(image_tensor)
    stego_score = output[0, 1]  # 'stego' class logit
    stego_score.backward()
    # Extract gradient for the first channel only
    grad = image_tensor.grad[0, 0].cpu().detach().numpy()
    return grad

def classify_image_direct(model, stego_array):
    """
    Classify the final stego image directly (with no extra preprocessing).
    Accepts either grayscale (H×W) or 3-channel (H×W×3) input.
    """
    if stego_array.ndim == 2:
        H, W = stego_array.shape
        rgb_array = np.stack([stego_array, stego_array, stego_array], axis=-1)
    else:
        rgb_array = stego_array
        H, W, _ = rgb_array.shape

    tensor_in = torch.tensor(rgb_array, dtype=torch.float32) / 255.0
    tensor_in = tensor_in.permute(2, 0, 1).unsqueeze(0)  # shape: (1,3,H,W)
    with torch.no_grad():
        output = model(tensor_in)
        pred = output.argmax(dim=1).item()  # 0 for COVER, 1 for STEGO
    return pred

############################################
# Part 2: EMD + Adversarial Embedding (Group of 3 Pixels)
############################################

def Extraction(v_img, C, n, g):
    """
    Extraction function for a group of n pixels.
      v_img: a list/array of pixel values (length n).
      C: coefficient list of length n.
      g: modulus.
    Computes: (sum_i C[i]*v_img[i]) mod g.
    """
    temp = 0
    for i in range(n):
        temp += v_img[i] * C[i]
    return int(temp) % g

def sign_matches(dx, grad_s):
    """
    Check whether a proposed change (dx) is in the OPPOSITE direction to the gradient.
      - If grad_s is positive, we require dx < 0.
      - If grad_s is negative, we require dx > 0.
      - If grad_s is '0' (near zero), any dx is acceptable.
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

def lookupT322(d):
    """
    Lookup function for a 3-pixel group.
    Given a difference d (computed as needed_diff = (s - (sum_i C[i]*group[i]) mod state)),
    return the corresponding change vector [dx1, dx2, dx3] according to the lookup table.

    (The lookup table below is taken from Code2’s Excel data.)
    """
    lookup_table = {
        14: [-2, -2, -1],
        54: [-2, -2, 0],
        94: [-2, -2, 1],
        79: [-2, -1, -2],
        19: [-2, -1, -1],
        59: [-2, -1, 0],
        99: [-2, -1, 1],
        39: [-2, -1, 2],
        84: [-2, 0, -2],
        24: [-2, 0, -1],
        64: [-2, 0, 0],
         4: [-2, 0, 1],
        44: [-2, 0, 2],
        89: [-2, 1, -2],
        29: [-2, 1, -1],
        69: [-2, 1, 0],
         9: [-2, 1, 1],
        49: [-2, 1, 2],
        34: [-2, 2, -1],
        74: [-2, 2, 0],
        32: [-1, -2, -1],
        72: [-1, -2, 0],
        12: [-1, -2, 1],
        97: [-1, -1, -2],
        37: [-1, -1, -1],
        77: [-1, -1, 0],
        17: [-1, -1, 1],
        57: [-1, -1, 2],
         2: [-1, 0, -2],
        42: [-1, 0, -1],
        82: [-1, 0, 0],
        22: [-1, 0, 1],
        62: [-1, 0, 2],
         7: [-1, 1, -2],
        47: [-1, 1, -1],
        87: [-1, 1, 0],
        27: [-1, 1, 1],
        67: [-1, 1, 2],
        52: [-1, 2, -1],
        92: [-1, 2, 0],
        90: [0, -2, 0],
        30: [0, -2, 1],
        15: [0, -1, -2],
        55: [0, -1, -1],
        95: [0, -1, 0],
        35: [0, -1, 1],
        75: [0, -1, 2],
        20: [0, 0, -2],
        60: [0, 0, -1],
         0: [0, 0, 0],
        40: [0, 0, 1],
        80: [0, 0, 2],
        25: [0, 1, -2],
        65: [0, 1, -1],
         5: [0, 1, 0],
        45: [0, 1, 1],
        85: [0, 1, 2],
        70: [0, 2, -1],
        10: [0, 2, 0],
        50: [0, 2, 1],
         8: [1, -2, 0],
        48: [1, -2, 1],
        33: [1, -1, -2],
        73: [1, -1, -1],
        13: [1, -1, 0],
        53: [1, -1, 1],
        93: [1, -1, 2],
        38: [1, 0, -2],
        78: [1, 0, -1],
        18: [1, 0, 0],
        58: [1, 0, 1],
        98: [1, 0, 2],
        43: [1, 1, -2],
        83: [1, 1, -1],
        23: [1, 1, 0],
        63: [1, 1, 1],
         3: [1, 1, 2],
        88: [1, 2, -1],
        28: [1, 2, 0],
        68: [1, 2, 1],
        26: [2, -2, 0],
        66: [2, -2, 1],
        51: [2, -1, -2],
        91: [2, -1, -1],
        31: [2, -1, 0],
        71: [2, -1, 1],
        11: [2, -1, 2],
        56: [2, 0, -2],
        96: [2, 0, -1],
        36: [2, 0, 0],
        76: [2, 0, 1],
        16: [2, 0, 2],
        61: [2, 1, -2],
         1: [2, 1, -1],
        41: [2, 1, 0],
        81: [2, 1, 1],
        21: [2, 1, 2],
         6: [2, 2, -1],
        46: [2, 2, 0],
        86: [2, 2, 1]
    }
    return lookup_table.get(d, [0, 0, 0])  # Default to no change if not found

def find_alternative_changes_n(x, s, grad_sign, C, mod, max_range=8):
    """
    Heuristic search (via recursive backtracking) for an alternative change vector dx (length n)
    such that:
       (sum_i C[i]*(x[i] + dx[i])) mod mod == s,
    and for each pixel i, the change dx[i] is within a candidate range and is in the opposite
    direction of the gradient (according to sign_matches).

    Returns the change vector dx if found, or None otherwise.
    """
    n = len(x)
    d_orig = sum(C[i]*x[i] for i in range(n)) % mod
    target = (s - d_orig) % mod
    best_solution = None
    best_cost = float('inf')

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

def embed_adversarial_emd(cover_array, gradient, m=102, C=[1, 13, 18]):
    """
    Embed using EMD with adversarial (gradient-based) adjustments, operating only on the first channel.
    Here the embedding is done on groups of 3 pixels.

    Parameters:
      cover_array: 2D NumPy array (grayscale image).
      gradient: gradient array (same shape as cover_array).
      m: modulus (set to 102).
      C: coefficient list (length 3, here [1, 13, 18]).

    Returns the final stego image (2D array, shape (H,W)) and prints embedding statistics.
    """
    H, W = cover_array.shape
    l = H * W
    n = 3  # group size is 3 pixels
    # Flatten image and gradient arrays.
    img_flat = cover_array.flatten().astype(np.float64)
    grad_flat = gradient.flatten()

    ls = l // n  # number of complete groups
    # Generate a random secret message for each group (each in 0...m-1)
    secret = np.random.randint(0, m, ls)

    count_already = 0
    count_align = 0
    count_heuristic = 0
    count_fallback = 0
    idx = 0
    for j in range(ls):
        group = img_flat[idx: idx+n]
        s_val = secret[j]
        # Check if the current group already extracts to the secret.
        curr_ext = Extraction(group, C, n, m)
        if curr_ext == s_val:
            count_already += 1
            idx += n
            continue

        # Compute the needed difference.
        sum_orig = sum(C[i]*group[i] for i in range(n)) % m
        needed_diff = (s_val - sum_orig) % m

        # Get default change from the lookup table.
        default_chg = lookupT322(needed_diff)

        # Get gradient signs for the group pixels.
        grad_signs = [get_grad_sign(grad_flat[idx+i]) for i in range(n)]

        # If the default change is adversarial for every pixel, use it.
        if all(sign_matches(default_chg[i], grad_signs[i]) for i in range(n)):
            new_group = group + np.array(default_chg)
            if np.all(new_group >= 0) and np.all(new_group <= 255):
                img_flat[idx: idx+n] = new_group
                count_align += 1
                idx += n
                continue

        # Otherwise, try to find an alternative adversarial change.
        alt_chg = find_alternative_changes_n(group, s_val, grad_signs, C, m)
        if alt_chg is None:
            # Fallback: use the default lookup change even if not adversarial.
            new_group = group + np.array(default_chg)
            if np.all(new_group >= 0) and np.all(new_group <= 255):
                img_flat[idx: idx+n] = new_group
            count_fallback += 1
        else:
            new_group = group + np.array(alt_chg)
            if np.all(new_group >= 0) and np.all(new_group <= 255):
                img_flat[idx: idx+n] = new_group
            count_heuristic += 1
        idx += n

    stego_array = img_flat.reshape(H, W).astype(np.uint8)

    # Print embedding statistics.
    print(f"Total pixel groups: {ls}")
    print(f"Groups already correct (no change needed): {count_already}")
    print(f"Groups using default adversarial change: {count_align}")
    print(f"Groups using heuristic alternative changes: {count_heuristic}")
    print(f"Groups using fallback default change: {count_fallback}")

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

    # Load the YeNet model.
    model = Model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # List all image files in the cover directory.
    image_files = [f for f in os.listdir(cover_dir)
                   if f.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg', '.bmp'))]

    for file_name in image_files:
        image_path = os.path.join(cover_dir, file_name)
        print(f"\nProcessing image: {image_path}")

        # Load the cover image and compute the gradient for channel 0.
        cover_img, cover_tensor = load_image_for_model(image_path)
        gradient_cover = calculate_gradient(model, cover_tensor)

        # Convert the cover image to a NumPy array.
        cover_array = np.array(cover_img, dtype=np.uint8)

        # Perform EMD embedding (only in the first channel) with the adversarial objective.
        # Here we use m=102 and C=[1, 13, 18] for 3-pixel groups.
        stego_array_1ch = embed_adversarial_emd(cover_array, gradient_cover, m=102, C=[1, 13, 18])

        # Create a 3-channel stego image for input to YeNet.
        H, W = stego_array_1ch.shape
        stego_3ch = np.stack([stego_array_1ch, stego_array_1ch, stego_array_1ch], axis=-1)

        # Classify the stego image.
        pred_label = classify_image_direct(model, stego_3ch)
        if pred_label == 0:
            print("YeNet classifies the final image as COVER.")
        else:
            print("YeNet classifies the final image as STEGO.")

        # Save the stego image.
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
