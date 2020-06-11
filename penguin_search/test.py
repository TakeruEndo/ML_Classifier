from albumentations.pytorch import ToTensorV2
import albumentations as albu
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch
import argparse
import cv2
import os
# おまじない
# https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


NAME_LABELS = {'EmperorPenguin': 1, 'RockhopperPenguin': 2, 'GalapagosPenguin': 3, 'KingPenguin': 4,
               'White-flipperedPenguin': 5, 'HumboldtPenguin': 6, 'MagellanicPenguin': 7, 'AfricanPenguin': 8, 'LittlePenguin': 9,
               'ChinstrapPenguin': 10, 'FiordlandPenguin': 11, 'Erect-crestedPenguin': 12, 'SnaresIslandsPenguin': 13, 'RoyalPenguin': 14,
               'MacaroniPenguin': 15, 'AdeliePenguin': 16, 'GentooPenguin': 17, 'Yellow-eyedPenguin': 18}


class PRDataset(Dataset):
    def __init__(self, paths, transforms=None):
        self.image_paths = []
        self.transforms = transforms
        for path in paths:
            print(path)
            self.image_paths.append(path)
        self.num_samples = len(self.image_paths)
        print(f"[Detection Dataset] size: {self.num_samples}")

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        print(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=image)
        image = augmented['image']
        return image, "None"

    def __len__(self):
        return self.num_samples


def load_net(checkpoint_path):
    model = models.resnext50_32x4d()
    print(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, torch.device('cpu')))
    model.eval()
    return model


def predict(model, data_loader, device):
    predict_list = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predict_list.append(preds[0])
    return predict_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', help='test image paths list')
    parser.add_argument('--model_path')
    args = parser.parse_args()

    model = load_net(args.model_path)

    img_height = 256
    img_width = 256

    # test用のデータ拡張
    data_transforms_test = albu.Compose([
        albu.Resize(img_height, img_width),
        albu.Normalize(),
        ToTensorV2()
    ])
    test_dataset = PRDataset([args.paths], transforms=data_transforms_test)
    num_workers = 0
    batch_size = 1
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=0)

    device = torch.device("cpu")
    label = predict(model, test_loader, device)

    def get_key_from_value(d, val):
        keys = [k for k, v in d.items() if v == val]
        if keys:
            return keys[0]
        return None

    print(label[0])
