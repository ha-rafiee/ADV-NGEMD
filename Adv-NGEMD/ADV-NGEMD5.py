#YeNet # O3 #one_Image
import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
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

    We'll convert it to torch format, but do not do any extra transformations
    except normalizing to [0,1].
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
        pred = output.argmax(dim=1).item()  # 0 or 1
    return pred

############################################
# Part 2: EMD + Adversarial Embedding (Only 1st Channel)
############################################

def lookupT212(d):
    """
    Original optimal lookup for difference d in base 9 for 2-pixel group.
    """
    table = {
        0: [0, 0],
        1: [1, 0],
        2: [-1, 1],
        3: [0, 1],
        4: [1, 1],
        5: [-1, -1],
        6: [0, -1],
        7: [1, -1],
        8: [-1, 0]
    }
    return table.get(d, [0, 0])

def Extraction(v_img, C, n, g):
    """
    EMD extraction function for group size n=2,
    coefficients C, modulus g.
    """
    temp = 0
    for i in range(n):
        temp += v_img[i]*C[i]
    return int(temp) % g

def sign_matches(dx, grad_s):
    """
    Check whether a proposed change (dx) is in the OPPOSITE direction of the gradient.
    That is:
      - If grad_s is '+' (i.e. gradient positive), then we require dx < 0.
      - If grad_s is '-' (i.e. gradient negative), then we require dx > 0.
      - If grad_s is '0', we accept any dx.
    """
    if grad_s == '+':
        return dx < 0
    elif grad_s == '-':
        return dx > 0
    else:
        return True

def get_grad_sign(g_val, threshold=1e-6):
    """
    Returns '+' if g_val>threshold, '-' if g_val<-threshold, else '0'.
    """
    if g_val > threshold:
        return '+'
    elif g_val < -threshold:
        return '-'
    else:
        return '0'

def find_alternative_changes(x1, x2, s, grad_sign, C=[1,3], mod=9):
    """
    Solve (c1*(x1+dx1) + c2*(x2+dx2)) mod mod = s with:
      - dx1, dx2 in [-8,8] (i.e. from -m+1 to m-1)
      - Only accept changes that are in the opposite direction of the gradient (using sign_matches)
      - Skip any candidate that exactly matches the default table solution.
      - Pick the solution with the minimal (|dx1| + |dx2|).
    """
    c1, c2 = C
    solutions = []

    # Compute default table solution so we can skip it
    sum_orig = (c1*x1 + c2*x2) % mod
    needed_diff = (s - sum_orig) % mod
    default_chg = lookupT212(needed_diff)

    for dx1 in range(-8, 8):
        for dx2 in range(-8, 8):
            # Enforce that the candidate changes are in the opposite direction of the gradient
            if not sign_matches(dx1, grad_sign[0]):
                continue
            if not sign_matches(dx2, grad_sign[1]):
                continue

            # Skip if it matches the default table pattern exactly
            if dx1 == default_chg[0] and dx2 == default_chg[1]:
                continue

            new_val = (c1*(x1+dx1) + c2*(x2+dx2)) % mod
            if new_val == s:
                cost = abs(dx1) + abs(dx2)
                solutions.append((dx1, dx2, cost))

    if not solutions:
        return None

    best = min(solutions, key=lambda x: x[2])
    return [best[0], best[1]]

def embed_adversarial_emd(cover_array, gradient, m=9, C=[1,3]):
    """
    Embed using EMD with adversarial (gradient-based) adjustments, only in the *first channel*.
    Returns the final stego image (2D, shape (H,W)) and prints:
      1. Total pixel groups,
      2. # groups that used default changes (when the default table solution is in the opposite
         direction of the gradient),
      3. # groups that used heuristic changes,
      4. # groups that used fallback default (when no heuristic solution is found).
    """
    H, W = cover_array.shape
    l = H * W
    n = 2

    # Flatten the image and gradient arrays.
    img_flat = cover_array.flatten().astype(np.float64)
    grad_flat = gradient.flatten()

    # Total pixel groups.
    ls = l // n

    # Generate a random secret message (values 0..m-1) for each group.
    secret = np.random.randint(0, m, ls)

    # For debugging: optionally count groups that already match.
    count_already = 0

    # Counters.
    total_groups = ls
    count_align = 0
    count_heuristic = 0
    count_fallback = 0

    idx = 0
    for j in range(ls):
        x1 = img_flat[idx]
        x2 = img_flat[idx+1]
        s_val = secret[j]

        # Check if current group already extracts to the secret.
        curr_ext = Extraction([x1, x2], C, n, m)
        if curr_ext == s_val:
            count_already += 1
            idx += 2
            continue

        # Compute default table change.
        sum_orig = (C[0]*x1 + C[1]*x2) % m
        needed_diff = (s_val - sum_orig) % m
        default_chg = lookupT212(needed_diff)

        # Get gradient signs for the two pixels.
        g1 = grad_flat[idx]
        g2 = grad_flat[idx+1]
        sign1 = get_grad_sign(g1)
        sign2 = get_grad_sign(g2)

        # Check if the default table changes are in the OPPOSITE direction to the gradient.
        # (Recall: sign_matches() returns True if the change is opposite to the gradient sign.)
        if sign_matches(default_chg[0], sign1) and sign_matches(default_chg[1], sign2):
            # The default change is already adversarial.
            new_p1 = x1 + default_chg[0]
            new_p2 = x2 + default_chg[1]
            if 0 <= new_p1 <= 255 and 0 <= new_p2 <= 255:
                img_flat[idx]   = new_p1
                img_flat[idx+1] = new_p2
                count_align += 1
        else:
            # Try to find alternative changes that are adversarial and yield the correct extraction.
            alt_chg = find_alternative_changes(x1, x2, s_val, (sign1, sign2), C=C, mod=m)
            if alt_chg is None:
                # Fallback: use the default table change even though it is not adversarial.
                new_p1 = x1 + default_chg[0]
                new_p2 = x2 + default_chg[1]
                if 0 <= new_p1 <= 255 and 0 <= new_p2 <= 255:
                    img_flat[idx]   = new_p1
                    img_flat[idx+1] = new_p2
                count_fallback += 1
            else:
                # Use the heuristic alternative change.
                new_p1 = x1 + alt_chg[0]
                new_p2 = x2 + alt_chg[1]
                if 0 <= new_p1 <= 255 and 0 <= new_p2 <= 255:
                    img_flat[idx]   = new_p1
                    img_flat[idx+1] = new_p2
                count_heuristic += 1

        idx += 2

    stego_array = img_flat.reshape(H, W).astype(np.uint8)

    # Print stats.
    print(f"Total pixel groups: {total_groups}")
    print(f"Groups already correct (no change needed): {count_already}")
    print(f"Groups used default changes (adversarial alignment): {count_align}")
    print(f"Groups used heuristic changes: {count_heuristic}")
    print(f"Groups used fallback default: {count_fallback}")

    return stego_array

############################################
# Part 3: Main Script
############################################

def main():
    # Example paths.
    cover_path = "/content/drive/MyDrive/Steganalysis_Project/testJSMA/cover/6994.pgm"
    checkpoint_path = "/content/drive/MyDrive/Steganalysis_Project/checkpoint_YeNet/checkpoint_100.pt"

    # 1) Load YeNet.
    model = Model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # 2) Load cover image & compute gradient for channel 0.
    cover_img, cover_tensor = load_image_for_model(cover_path)
    gradient_cover = calculate_gradient(model, cover_tensor)

    # 'cover_img' is a 2D grayscale. We'll embed only in that array.
    cover_array = np.array(cover_img, dtype=np.uint8)

    # 3) Perform EMD embedding (in 1st channel) aligned with the adversarial objective.
    stego_array_1ch = embed_adversarial_emd(cover_array, gradient_cover, m=9, C=[1,3])

    # 4) Create a final 3-channel image for YeNet.
    H, W = stego_array_1ch.shape
    stego_3ch = np.stack([stego_array_1ch, stego_array_1ch, stego_array_1ch], axis=-1)

    # 5) Classify the final stego image with no extra preprocessing.
    pred_label = classify_image_direct(model, stego_3ch)
    if pred_label == 0:
        print("YeNet classifies the final image as COVER.")
    else:
        print("YeNet classifies the final image as STEGO.")

    # 6) Save the final stego image.
    out_dir = os.path.dirname(cover_path)
    out_name = f"advEMD_{os.path.basename(cover_path)}"
    out_path = os.path.join(out_dir, out_name)
    Image.fromarray(stego_array_1ch).save(out_path)
    print(f"Stego image saved to: {out_path}")

    psnr_value = peak_signal_noise_ratio(cover_array, stego_array_1ch, data_range=255)
    ssim_value = structural_similarity(cover_array, stego_array_1ch, data_range=255)
    print(f"PSNR (Cover vs Stego): {psnr_value}")
    print(f"SSIM (Cover vs Stego): {ssim_value}")


if __name__ == "__main__":
    main()
