import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from models.SRNet import Model
import config as c
import random
import pickle
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def get_model_path():
    """Get the correct path to the pre-trained model"""
    model_path = '/content/drive/MyDrive/XuNet_Project/checkpoint_Yenet/checkpoint_100.pt'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found at: {model_path}")

    return model_path

def check_paths():
    """Verify all required paths exist"""
    required_paths = {
        'model': '/content/drive/MyDrive/XuNet_Project/checkpoint_Yenet/checkpoint_100.pt',
        'cover_image': "/content/drive/MyDrive/XuNet_Project/testJSMA/cover/1.pgm",
        'output_dir': "/content/drive/MyDrive/XuNet_Project/testJSMA/stego_results"
    }

    for name, path in required_paths.items():
        if not os.path.exists(path) and name != 'output_dir':
            raise FileNotFoundError(f"{name} not found at: {path}")

    # Create output directory if it doesn't exist
    os.makedirs(required_paths['output_dir'], exist_ok=True)

    return required_paths

class GroupStego:
    def __init__(self):
        self.n = 2  # group size
        self.state = 9  # number of states
        self.C = np.array([1, 3])  # coefficient array

    def lookupT212(self, d):
        lookup = {
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
        return lookup.get(d, [0, 0])

    def extraction(self, v_img):
        temp = 0
        for i in range(self.n):
            temp += v_img[i] * self.C[i]
        return temp % self.state

class StegoAttack:
    def __init__(self, model, epsilon=0.3, alpha=0.05, num_iterations=100):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.CrossEntropyLoss()
        self.group_stego = GroupStego()

    def identify_sensitive_pixels(self, image):
        """Find pixels that influence model's classification using iterative gradient analysis"""
        image = image.clone().detach().requires_grad_(True)
        target = torch.zeros(1, dtype=torch.long).to(self.device)

        grad_accumulator = torch.zeros_like(image)
        print("Starting gradient analysis...")

        for i in range(self.num_iterations):
            if image.grad is not None:
                image.grad.zero_()

            output = self.model(image)
            probs = F.softmax(output, dim=1)
            loss = -torch.log(probs[0, 0]) + torch.log(probs[0, 1])

            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss.item():.6f}")

            loss.backward(retain_graph=True)

            if image.grad is None or torch.all(image.grad == 0):
                print(f"Warning: Zero gradients at iteration {i}")
                continue

            with torch.no_grad():
                current_grad = torch.abs(image.grad)
                grad_accumulator += current_grad / torch.max(current_grad)

                perturb = self.alpha * image.grad.sign()
                image = image + perturb
                image = torch.clamp(image, 0, 1)
                image.requires_grad_(True)

        grad_numpy = grad_accumulator.squeeze().cpu().numpy()
        threshold = np.percentile(grad_numpy, 70)

        max_grad = grad_numpy.max()
        min_grad = grad_numpy.min()

        if max_grad > min_grad:
            grad_normalized = (grad_numpy - min_grad) / (max_grad - min_grad)
        else:
            print("Warning: Uniform gradients detected")
            grad_normalized = np.abs(grad_numpy)

        significant_pixels = []

        for c in range(grad_numpy.shape[0]):
            for y in range(grad_numpy.shape[1]):
                for x in range(grad_numpy.shape[2]):
                    if grad_normalized[c, y, x] > 0.1:
                        influence_score = grad_normalized[c, y, x]
                        direction = 1 if image.grad[0, c, y, x] > 0 else -1

                        significant_pixels.append({
                            'position': (x, y),
                            'channel': c,
                            'gradient': influence_score,
                            'direction': direction,
                            'score': influence_score * abs(direction)
                        })

        significant_pixels.sort(key=lambda x: x['score'], reverse=True)

        if len(significant_pixels) == 0:
            print("\nFalling back to basic pixel selection...")
            for c in range(grad_numpy.shape[0]):
                for y in range(0, grad_numpy.shape[1], 4):
                    for x in range(0, grad_numpy.shape[2], 4):
                        significant_pixels.append({
                            'position': (x, y),
                            'channel': c,
                            'gradient': 1.0,
                            'direction': 1,
                            'score': 1.0
                        })

        return significant_pixels

    def group_sensitive_pixels(self, sensitive_pixels, image_shape):
        """Group sensitive pixels into pairs for group-based embedding"""
        height, width = image_shape[2], image_shape[3]
        pixel_groups = []
        used_positions = set()  # Track used positions instead of indices

        # Sort pixels by row first, then column
        sensitive_pixels.sort(key=lambda x: (x['position'][1], x['position'][0]))

        i = 0
        while i < len(sensitive_pixels) - 1:
            pixel1 = sensitive_pixels[i]
            pos1 = pixel1['position']

            # Skip if position already used
            if pos1 in used_positions:
                i += 1
                continue

            x1, y1 = pos1
            found_pair = False

            # Look for next available pixel in same row
            for j in range(i + 1, len(sensitive_pixels)):
                pixel2 = sensitive_pixels[j]
                pos2 = pixel2['position']

                # Skip if position already used
                if pos2 in used_positions:
                    continue

                x2, y2 = pos2

                # Check if pixels are in same row and properly spaced
                if y1 == y2 and x1 != x2:  # Must be different x positions
                    distance = abs(x2 - x1)
                    if distance <= 8:  # Maximum distance of 8 pixels
                        # Create group
                        group = {
                            'positions': [pos1, pos2],
                            'channels': [pixel1['channel'], pixel2['channel']],
                            'gradients': [pixel1['gradient'], pixel2['gradient']],
                            'directions': [pixel1['direction'], pixel2['direction']],
                            'score': (pixel1['score'] + pixel2['score']) / 2
                        }
                        pixel_groups.append(group)
                        used_positions.add(pos1)
                        used_positions.add(pos2)
                        found_pair = True
                        i = j + 1  # Move past both used pixels
                        break

            if not found_pair:
                i += 1

        print(f"Created {len(pixel_groups)} pixel groups from {len(sensitive_pixels)} sensitive pixels")

        # Debug output for first few groups
        print("\nSample groups:")
        for i, group in enumerate(pixel_groups[:300]):
            print(f"Group {i+1}:")
            print(f"  Positions: {group['positions']}")
            print(f"  Score: {group['score']:.6f}")

        return pixel_groups
def load_and_preprocess_image(image_path, img_size=512):
    """Load and preprocess cover image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert to 3 channels
    ])

    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def generate_random_message(length=200):
    """Generate random message for group-based embedding"""
    return [random.randint(0, 8) for _ in range(length)]  # 0-8 for 9 states
def embed_message_group(image, message_bits, pixel_groups):
    """Embed message using group-based method in sensitive pixel groups"""
    if not pixel_groups:
        raise ValueError("No pixel groups found!")

    stego_image = image.clone()
    embedded_count = 0
    embedded_groups = []
    group_stego = GroupStego()

    # Only process as many bits as we have groups for
    processable_bits = message_bits[:len(pixel_groups)]

    for i, (bit, group) in enumerate(zip(processable_bits, pixel_groups)):
        # Get pixel values from the image using positions and channels
        pixel_values = []
        positions = group['positions']
        channels = group['channels']

        # Read current pixel values
        for idx, (x, y) in enumerate(positions):
            c = channels[idx]
            pixel_values.append(int(round(float(image[0, c, y, x].item() * 255))))

        # Skip if any pixel value is >= 254
        if any(val >= 254 for val in pixel_values):
            continue

        # Calculate current extraction value
        current_ext = group_stego.extraction(np.array(pixel_values))

        # Calculate needed changes
        det = (bit - current_ext) % group_stego.state
        changes = group_stego.lookupT212(det)

        # Apply changes if within valid range
        new_values = [
            pixel_values[0] + changes[0],
            pixel_values[1] + changes[1]
        ]

        # Verify changes are valid
        if all(0 <= val < 255 for val in new_values):
            # Apply changes to image
            for idx, (x, y) in enumerate(positions):
                c = channels[idx]
                stego_image[0, c, y, x] = new_values[idx] / 255.0

            embedded_count += 1

            # Store embedding information
            embedded_groups.append({
                'positions': positions,
                'channels': channels,
                'original_values': pixel_values,
                'new_values': new_values,
                'original_bit': bit,
                'changes': changes,
                'index': i
            })

    print(f"Embedded {embedded_count} symbols out of {len(message_bits)}")
    return stego_image, embedded_groups

def extract_message_group(stego_image, embedded_groups):
    """Extract message using group-based method"""
    extracted_bits = []
    group_stego = GroupStego()

    for group in embedded_groups:
        pixel_values = []
        for idx, (x, y) in enumerate(group['positions']):
            c = group['channels'][idx]
            # Scale to [0,255] range for extraction
            pixel_values.append(int(round(float(stego_image[0, c, y, x].item() * 255))))

        extracted_bit = group_stego.extraction(np.array(pixel_values))
        extracted_bits.append(extracted_bit)

    return extracted_bits

def verify_message(original_bits, extracted_bits):
    """Compare original and extracted messages"""
    # Get number of actually embedded/extracted bits
    embedded_length = len(extracted_bits)

    # Only compare the bits that were actually embedded
    original_bits = original_bits[:embedded_length]

    if embedded_length == 0:
        return {
            'total_bits': 0,
            'correct_bits': 0,
            'accuracy': 0.0,
            'detailed_comparison': []
        }

    correct_bits = sum(1 for a, b in zip(original_bits, extracted_bits) if a == b)
    accuracy = (correct_bits / embedded_length) * 100

    # Show sample of original and extracted bits for verification
    print("\nMessage Comparison Sample (first 20 bits):")
    print("Original:", original_bits[:3000])
    print("Extracted:", extracted_bits[:3000])

    comparison = []
    compare_length = min(20, embedded_length)
    for i in range(compare_length):
        comparison.append({
            'index': i,
            'original': original_bits[i],
            'extracted': extracted_bits[i],
            'match': original_bits[i] == extracted_bits[i]
        })

    return {
        'total_bits': embedded_length,
        'correct_bits': correct_bits,
        'accuracy': accuracy,
        'detailed_comparison': comparison
    }

def calculate_metrics(cover_image, stego_image):
    """Calculate image quality metrics"""
    cover_np = cover_image.squeeze()[0].cpu().numpy()
    stego_np = stego_image.squeeze()[0].cpu().numpy()

    ps = psnr(cover_np, stego_np)
    ss = ssim(cover_np, stego_np, data_range=cover_np.max() - cover_np.min())

    return {
        'psnr': ps,
        'ssim': ss
    }
def save_results(save_dir, stego_image, embedded_groups, message_bits, metrics,
                verification, initial_probs, final_probs, cover_image_path,
                sensitive_pixels, pixel_groups):
    """Save all results and metrics"""
    os.makedirs(save_dir, exist_ok=True)

    # Save stego image
    stego_path = os.path.join(save_dir, 'stego_output.pgm')
    stego_np = stego_image.squeeze()[0].cpu().numpy()
    stego_save = (stego_np * 255).astype(np.uint8)
    Image.fromarray(stego_save).save(stego_path)

    # Save detailed results
    results_path = os.path.join(save_dir, 'stego_results.txt')
    with open(results_path, 'w') as f:
        f.write("Steganography Attack Results\n")
        f.write("==========================\n\n")

        f.write("1. Image Information\n")
        f.write(f"Cover image: {cover_image_path}\n")
        f.write(f"Output stego image: {stego_path}\n\n")

        f.write("2. Pixel Statistics\n")
        f.write(f"Total sensitive pixels found: {len(sensitive_pixels)}\n")
        f.write(f"Pixel groups created: {len(pixel_groups)}\n")
        f.write(f"Groups used for embedding: {len(embedded_groups)}\n\n")

        f.write("3. Embedding Statistics\n")
        f.write(f"Message length: {len(message_bits)}\n")
        f.write(f"Successfully embedded symbols: {len(embedded_groups)}\n\n")

        f.write("4. Quality Metrics\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
        f.write(f"SSIM: {metrics['ssim']:.4f}\n\n")

        f.write("5. Extraction Results\n")
        f.write(f"Total symbols processed: {verification['total_bits']}\n")
        f.write(f"Correctly extracted symbols: {verification['correct_bits']}\n")
        f.write(f"Extraction accuracy: {verification['accuracy']:.2f}%\n\n")

        f.write("6. Classification Results\n")
        f.write(f"Original - Cover prob: {initial_probs[0, 0]:.4f}, "
               f"Stego prob: {initial_probs[0, 1]:.4f}\n")
        f.write(f"Final    - Cover prob: {final_probs[0, 0]:.4f}, "
               f"Stego prob: {final_probs[0, 1]:.4f}\n\n")

        f.write("7. First 20 Symbols Comparison\n")
        for comp in verification['detailed_comparison']:
            f.write(f"Symbol {comp['index']}: Orig={comp['original']}, "
                   f"Extr={comp['extracted']}, "
                   f"{'Match' if comp['match'] else 'Mismatch'}\n")

        f.write("\n8. Sample Pixel Groups\n")
        for i, group in enumerate(pixel_groups[:5]):
            f.write(f"\nGroup {i+1}:\n")
            f.write(f"  Positions: {group['positions']}\n")
            f.write(f"  Channels: {group['channels']}\n")
            f.write(f"  Score: {group['score']:.6f}\n")

    # Save embedding details
    positions_path = os.path.join(save_dir, 'embedding_positions.pkl')
    with open(positions_path, 'wb') as f:
        pickle.dump({
            'sensitive_pixels': sensitive_pixels,
            'pixel_groups': pixel_groups,
            'embedded_groups': embedded_groups,
            'message_bits': message_bits,
            'metrics': metrics
        }, f)

    return stego_path, results_path, positions_path

def main():
    # Setup device and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Check all paths first
        paths = check_paths()
        print("All required paths verified.")

        # Load SRNet model
        print("Loading SRNet model...")
        model = Model().to(device)
        model.load_state_dict(torch.load(paths['model'], map_location=device))
        model.eval()

        # Initialize attack
        attack = StegoAttack(
            model,
            epsilon=0.3,
            alpha=0.05,
            num_iterations=100
        )

        # Load cover image
        print(f"Loading cover image from: {paths['cover_image']}")
        cover_image = load_and_preprocess_image(paths['cover_image']).to(device)
        print(f"Cover image shape: {cover_image.shape}")

        # Initial classification check
        with torch.no_grad():
            initial_output = model(cover_image)
            initial_probs = F.softmax(initial_output, dim=1)
        print(f"\nInitial classification:")
        print(f"Cover probability: {initial_probs[0, 0]:.4f}")
        print(f"Stego probability: {initial_probs[0, 1]:.4f}")

        # Find sensitive pixels
        print("\nIdentifying sensitive pixels...")
        sensitive_pixels = attack.identify_sensitive_pixels(cover_image)
        print(f"\nFound {len(sensitive_pixels)} sensitive pixels")

        if len(sensitive_pixels) > 0:
            # Print sample of sensitive pixels
            print("\nSample of sensitive pixels:")
            for i, pixel in enumerate(sensitive_pixels[:5]):
                print(f"Pixel {i+1}:")
                print(f"  Position: {pixel['position']}")
                print(f"  Channel: {pixel['channel']}")
                print(f"  Gradient: {pixel['gradient']:.6f}")
                print(f"  Score: {pixel['score']:.6f}")

            # Group sensitive pixels
            print("\nGrouping sensitive pixels...")
            pixel_groups = attack.group_sensitive_pixels(sensitive_pixels, cover_image.shape)

            # Generate random message
            message_length = min(len(pixel_groups), 5000)
            message_bits = generate_random_message(message_length)
            print(f"\nGenerated random message of length: {len(message_bits)}")

            # Embed message
            print("\nPerforming embedding...")
            stego_image, embedded_groups = embed_message_group(cover_image, message_bits, pixel_groups)
            stego_image = stego_image.to(device)

            # Extract and verify message
            print("\nExtracting message...")
            extracted_bits = extract_message_group(stego_image.cpu(), embedded_groups)

            # Verify extraction
            verification = verify_message(message_bits, extracted_bits)

            # Calculate metrics
            metrics = calculate_metrics(cover_image, stego_image)

            print("\nSteganography Results:")
            print(f"PSNR: {metrics['psnr']:.2f} dB")
            print(f"SSIM: {metrics['ssim']:.4f}")
            print(f"Extraction accuracy: {verification['accuracy']:.2f}%")
            print(f"Total symbols embedded: {verification['total_bits']}")
            print(f"Correctly extracted symbols: {verification['correct_bits']}")

            # Check final classification
            with torch.no_grad():
                final_output = model(stego_image)
                final_probs = F.softmax(final_output, dim=1)

            print(f"\nFinal classification:")
            print(f"Cover probability: {final_probs[0, 0]:.4f}")
            print(f"Stego probability: {final_probs[0, 1]:.4f}")

            # Save results
            stego_path, results_path, positions_path = save_results(
                paths['output_dir'],
                stego_image,
                embedded_groups,
                message_bits,
                metrics,
                verification,
                initial_probs,
                final_probs,
                paths['cover_image'],
                sensitive_pixels,
                pixel_groups
            )

            print("\nSaved files:")
            print(f"Stego image: {stego_path}")
            print(f"Results file: {results_path}")
            print(f"Embedding positions: {positions_path}")

        else:
            print("\nWarning: Failed to find sensitive pixels.")
            print("Try adjusting parameters.")

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()