from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

root = Path('data/food11_processed_mini')
normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), normalize])
loader = DataLoader(datasets.ImageFolder(root / 'training', transform=transform), batch_size=8, shuffle=True, num_workers=0)
print('before first batch')
b = next(iter(loader))
print('first batch shape', b[0].shape)
