import torch
from process_data import get_data_transforms, SIIM_ISIC
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

device = 'cuda:2'

''''
author: Hu Shengran
'''

def calculate_mean_std():
    train_transform, valid_transform = get_data_transforms()
    # trainset = SIIM_ISIC(transform=train_transform)
    # loader = torch.utils.data.DataLoader(
    #     trainset,
    #     batch_size=4,
    #     num_workers=4,
    #     shuffle=False
    # )

    validset = SIIM_ISIC(train=False, transform=valid_transform)

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=4,
        num_workers=4,
        shuffle=True
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    # for inputs, meta, targets in loader:
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     nb_samples += 1
    #     if nb_samples % 10 == 0:
    #         print('step: ', nb_samples)

    for inputs, meta, targets in validloader:
        inputs, targets = inputs.to(device), targets.to(device)
        nb_samples += 1
        if nb_samples % 10 == 0:
            print('step: ', nb_samples)

    mean /= nb_samples
    std /= nb_samples
    print('mean: ', mean, 'std: ', std)

''''
author: Hu Shengran
'''
if __name__ == '__main__':
    train_transform, valid_transform = get_data_transforms()
    trainset = SIIM_ISIC(transform=train_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2)

    validset = SIIM_ISIC(train=False, transform=valid_transform)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=4, shuffle=True, num_workers=2)

    testset = SIIM_ISIC(train=False, test=True, transform=valid_transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=True, num_workers=16)


    # how transform example
    # image = Image.open('/home/group3/DataSet/Training_set/img_1.jpg')
    # ori_img = transforms.ToTensor()(image)
    # image = train_transform(image)
    # plt.imshow(image.permute(1, 2, 0))
    # plt.show()
    # plt.imshow(ori_img.permute(1, 2, 0))
    # plt.show()

    # calculate mean_std
    #calculate_mean_std()

    for batch_idx, (inputs, meta) in enumerate(testloader):
        inputs = inputs.to(device)
        sex = meta['sex']
        age = meta['age_approx']
        location = meta['anatom_site_general_challenge']
        print(batch_idx)
        # training process
        pass
