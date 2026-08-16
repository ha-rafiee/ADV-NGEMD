#SRNet/YeNet
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
#from models.SRNet import Model
#from models.StegNet import Model, initWeights
from models.YeNet import Model
def load_image(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.repeat(1, 3, 1, 1)
    return img_tensor

def calculate_gradient(model, image_tensor):
    image_tensor.requires_grad = True
    output = model(image_tensor)
    stego_score = output[0, 1]
    stego_score.backward()
    gradient = image_tensor.grad[0, 0].cpu().detach().numpy()
    return gradient

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
    greater_20_count = np.sum(gradient > 30)
    less_neg_20_count = np.sum(gradient < -30)
    stats = {
        "min_non_zero": np.min(non_zero_gradients),
        "max_non_zero": np.max(non_zero_gradients),
        "mean_abs_non_zero": mean_abs_non_zero,
        "mean_abs_greater_0001": mean_abs_greater_0001,
        "mean_abs_greater_1": mean_abs_greater_1,
        "greater_0001_count": greater_0001_count,
        "less_neg_0001_count": less_neg_0001_count,
        "greater_20_count": greater_20_count,
        "less_neg_20_count": less_neg_20_count,
    }
    return stats

if __name__ == "__main__":
    image_path = "/content/drive/MyDrive/Steganalysis_Project/testJSMA/cover/6991.pgm"
    #checkpoint_path = "/content/drive/MyDrive/Steganalysis_Project/checkpoint_SRNet/checkpoint_100.pt"
    #checkpoint_path = '/content/drive/MyDrive/Steganalysis_Project/checkpoint_StegNet/checkpoint_100.pt'
    checkpoint_path = '/content/drive/MyDrive/Steganalysis_Project/checkpoint_YeNet/checkpoint_100.pt'

    model = Model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    image_tensor = load_image(image_path)
    gradient = calculate_gradient(model, image_tensor)

    stats = analyze_gradients(gradient)
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")
