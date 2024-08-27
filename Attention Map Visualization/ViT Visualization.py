import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from timm import create_model

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.ablation_layer import AblationLayerVit


class ViT_B16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vitb16 = create_model('vit_base_patch16_224', pretrained=True)
        self.vitb16.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.vitb16(x)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png', help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle component of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam', help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

def reshape_transform(tensor):
    result = tensor[:, 1:, :]
    num_patches = result.size(1)
    height = width = int(num_patches ** 0.5)
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

if __name__ == '__main__':
    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad
    }

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    device = torch.device("cuda" if args.use_cuda else "cpu")

    vit_B8_model = ViT_B16(num_classes=3)
    state_dict = torch.load("/home/ahmed/PycharmProjects/Covid/BEST_MODELS_IN_LUNG/lung_cancer_Best_models/Original/ViTB16/best_model.pth", map_location=device)

    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    vit_B8_model.load_state_dict(new_state_dict)
    vit_B8_model.to(device)
    vit_B8_model.eval()

    target_layers = [vit_B8_model.vitb16.blocks[-1].norm1]

    cam = methods[args.method](model=vit_B8_model, target_layers=target_layers, reshape_transform=reshape_transform)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    targets = [ClassifierOutputTarget(2)]

    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    grayscale_cam = grayscale_cam[0, :]

    rgb_img = np.array(image.resize((224, 224))) / 255
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)