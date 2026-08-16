# YeNet adversarial EMD embedding and detailed extraction logging
# #YeNet #O3 #one_Image

import torch
import numpy as np
import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.YeNet import Model  # Adjust import as needed

############################################
# Part 1: Gradient Calculation & Classification
############################################

def load_image_for_model(image_path):
    """
    Load grayscale image, normalize to [0,1], repeat to 3 channels for YeNet.
    Returns PIL image and torch tensor (1,3,H,W).
    """
    img = Image.open(image_path).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    tensor = tensor.repeat(1, 3, 1, 1)                     # (1,3,H,W)
    return img, tensor


def calculate_gradient(model, image_tensor):
    """
    Compute gradient of YeNet's stego logit (index 1) w.r.t. channel 0.
    """
    image_tensor.requires_grad = True
    output = model(image_tensor)
    stego_logit = output[0, 1]
    stego_logit.backward()
    grad = image_tensor.grad[0, 0].cpu().detach().numpy()
    return grad


def classify_image_direct(model, stego_array):
    """
    Classify an image array (H,W) or (H,W,3) with YeNet, normalizing to [0,1].
    Returns 0 for cover, 1 for stego.
    """
    if stego_array.ndim == 2:
        rgb = np.stack([stego_array]*3, axis=-1)
    else:
        rgb = stego_array
    tensor = torch.tensor(rgb, dtype=torch.float32) / 255.0
    tensor = tensor.permute(2,0,1).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        return out.argmax(dim=1).item()

############################################
# Part 2: EMD Helper Functions
############################################

def lookupT212(d):
    # Optimal 2-pixel EMD for base-9
    table = {
        0: [0,0], 1: [1,0], 2: [-1,1],
        3: [0,1], 4: [1,1], 5: [-1,-1],
        6: [0,-1],7: [1,-1],8: [-1,0]
    }
    return table.get(d, [0,0])


def Extraction(v, C, g):
    # Extract symbol from v (length=2) with coeffs C, modulus g
    return int((v[0]*C[0] + v[1]*C[1]) % g)


def get_grad_sign(val, thr=1e-6):
    if val > thr: return '+'
    if val < -thr: return '-'
    return '0'


def sign_matches(dx, sign):
    if sign == '+': return dx < 0
    if sign == '-': return dx > 0
    return True


def find_alternative_changes(x1, x2, target, sign1, sign2, C, g):
    c1, c2 = C
    orig = (c1*x1 + c2*x2) % g
    diff = int((target - orig) % g)
    default = lookupT212(diff)
    best = None
    best_cost = float('inf')
    for dx1 in range(-8,9):
        if not sign_matches(dx1, sign1): continue
        for dx2 in range(-8,9):
            if not sign_matches(dx2, sign2): continue
            if [dx1,dx2] == default: continue
            if ((c1*(x1+dx1) + c2*(x2+dx2)) % g) == target:
                cost = abs(dx1)+abs(dx2)
                if cost < best_cost:
                    best_cost, best = cost, [dx1,dx2]
    return best

############################################
# Part 3: Embedding with detailed change log
############################################

def embed_adversarial_emd(cover, gradient, secret, g=9, C=[1,3]):
    """
    Embed `secret` array (len = floor(H*W/2)).
    Returns:
      - stego_array (H,W) uint8
      - change_log: dict {group_index: change_type}
    change_type in {'no_change','align','heuristic','fallback'}
    """
    H, W = cover.shape
    flat = cover.flatten().astype(np.int16)
    grad = gradient.flatten()
    n = 2
    groups = len(flat)//n
    change_log = {}

    for j in range(groups):
        i = 2*j
        x1, x2 = int(flat[i]), int(flat[i+1])
        s = int(secret[j])
        orig = Extraction([x1,x2], C, g)
        if orig == s:
            change_log[j] = 'no_change'
            continue
        # compute default and signs
        diff = (s - orig) % g
        def_dx = lookupT212(diff)
        sign1 = get_grad_sign(grad[i])
        sign2 = get_grad_sign(grad[i+1])
        # choose change
        if sign_matches(def_dx[0], sign1) and sign_matches(def_dx[1], sign2):
            dx1, dx2 = def_dx
            change_log[j] = 'align'
        else:
            alt = find_alternative_changes(x1, x2, s, sign1, sign2, C, g)
            if alt is None:
                dx1, dx2 = def_dx
                change_log[j] = 'fallback'
            else:
                dx1, dx2 = alt
                change_log[j] = 'heuristic'
        # apply and clamp
        flat[i]   = np.clip(x1 + dx1, 0, 255)
        flat[i+1] = np.clip(x2 + dx2, 0, 255)

    stego = flat.reshape(H, W).astype(np.uint8)
    return stego, change_log

############################################
# Part 4: Extraction of embedded secret
############################################

def extract_adversarial_emd(stego, secret_len, C=[1,3], g=9):
    """
    Extract full message of length secret_len from stego.
    Returns dict {group_index: extracted_symbol} for all groups.
    """
    flat = stego.flatten().astype(np.int16)
    extracted = {}
    for j in range(secret_len):
        i = 2*j
        extracted[j] = Extraction([int(flat[i]), int(flat[i+1])], C, g)
    return extracted

############################################
# Part 5: Main Workflow with mismatch details
############################################

def main():
    cover_path      = "/content/drive/MyDrive/Steganalysis_Project/testJSMA/cover/6994.pgm"
    checkpoint_path = "/content/drive/MyDrive/Steganalysis_Project/checkpoint_YeNet/checkpoint_100.pt"

    # Load model
    model = Model()
    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()

    # Load & gradient
    cover_img, cover_t = load_image_for_model(cover_path)
    grad = calculate_gradient(model, cover_t)
    cover_arr = np.array(cover_img, dtype=np.uint8)

    # Prepare secret
    total_groups = cover_arr.size // 2
    secret = np.random.randint(0, 9, size=total_groups)

    # Embed with change log
    stego_arr, change_log = embed_adversarial_emd(cover_arr, grad, secret, g=9, C=[1,3])

    # Extract full message
    extracted = extract_adversarial_emd(stego_arr, total_groups, C=[1,3], g=9)

    # Compare and report mismatches with change types
    mismatches = 0
    for j in range(total_groups):
        if extracted[j] != int(secret[j]):
            mismatches += 1
            print(f"Group {j}: Secret={secret[j]}, Extracted={extracted[j]}, ChangeType={change_log[j]}")
    print(f"Total mismatched groups: {mismatches}")

    # Classification
    stego_3ch = np.stack([stego_arr]*3, axis=-1)
    label     = classify_image_direct(model, stego_3ch)
    print("YeNet → COVER" if label==0 else "YeNet → STEGO")

    # Save & metrics
    out_dir  = os.path.dirname(cover_path)
    out_file = os.path.join(out_dir, f"advEMD_{os.path.basename(cover_path)}")
    Image.fromarray(stego_arr).save(out_file)
    print(f"Saved stego: {out_file}")
    psnr = peak_signal_noise_ratio(cover_arr, stego_arr, data_range=255)
    ssim= structural_similarity(cover_arr, stego_arr, data_range=255)
    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

if __name__ == "__main__":
    main()
