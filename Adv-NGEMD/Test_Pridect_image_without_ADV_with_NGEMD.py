# test_for_predict 1 image_Yenet
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from models.YeNet import Model

def load_and_preprocess_image(image_path, img_size=512):
    """Load and preprocess image for the CNN model"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Ensure the image is in RGB format (3 channels)
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Removed the Lambda function
    ])

    image_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 512, 512]
    return image_tensor

def test_image(image_path, model_path):
    """Test if an image is cover or stego"""
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load and prepare image
        print(f"Loading image from: {image_path}")
        image_tensor = load_and_preprocess_image(image_path).to(device)
        print(f"Image tensor shape: {image_tensor.shape}")

        # Load model
        print("Loading model...")
        model = Model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Test image
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)

        # Print results
        print("\nClassification Results:")
        print(f"Cover probability: {probs[0, 0]:.4f}")
        print(f"Stego probability: {probs[0, 1]:.4f}")

        prediction = "Cover" if probs[0, 0] > probs[0, 1] else "Stego"
        print(f"\nFinal prediction: {prediction}")

        return {
            'cover_prob': probs[0, 0].item(),
            'stego_prob': probs[0, 1].item(),
            'prediction': prediction
        }

    except Exception as e:
        print(f"Error during testing: {e}")
        raise

def main():
    # Example usage
    image_path = "/content/drive/MyDrive/Steganalysis_Project/Paper3_Test/Yenet_adv_pgd/adv_4.pgm"
    model_path = "/content/drive/MyDrive/Steganalysis_Project/checkpoint_YeNet/checkpoint_100.pt"
    #image_path = "/content/drive/MyDrive/XuNet_Project/testJSMA/stego/stego_adv_pgd/stego_adv_s_PGD.pgm"
    results = test_image(image_path, model_path)
    print("\nResults:", results)

if __name__ == "__main__":
    main()
