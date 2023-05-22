import numpy as np
import torch as tc
import torchvision as tv
import SimpleITK as sitk

def predict_image(img_path):
    model_path = 'eff1.pth' # path do wag modelu

    model = tv.models.efficientnet_b0()

    model.features[0][0] = tc.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = tc.nn.Linear(1280,2,bias= True)

    model.load_state_dict(tc.load(model_path))

    image_path = img_path
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    transforms = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Resize((224,224)),
                tv.transforms.Normalize(mean=[0.485], std=[0.229])
                ])
    image = transforms(image)

    output = model(image.unsqueeze(0))
    prediction = tc.argmax(tc.nn.Sigmoid()(output), dim=1)
    return prediction.item()