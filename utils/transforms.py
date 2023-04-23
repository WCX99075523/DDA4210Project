import torchvision.transforms as transforms

# def get_default_transforms(size=256):
#     return transforms.Compose([
#         transforms.Resize(size=size),
#         transforms.CenterCrop(size=size),
#         transforms.RandomHorizontalFlip(p=.5),
#         transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
#     ])
def get_default_transforms(H=256,W=256):
    return transforms.Compose([
        transforms.Resize(size=(H,W)),
        transforms.CenterCrop(size=(H,W)),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])