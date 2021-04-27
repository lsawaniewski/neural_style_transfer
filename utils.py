import torch
from torch import optim
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt

from model import VGG19, get_style_model_and_losses, device

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

# normalisation used for vgg19 training
CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225]).to(device)


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(content_img, style_img, input_img,
                       num_steps=300, style_weight=1000000, content_weight=1,
                       cnn=VGG19, normalization_mean=CNN_NORMALIZATION_MEAN, normalization_std=CNN_NORMALIZATION_STD):
    """Run the style transfer."""

    assert style_img.size() == content_img.size() == input_img.size(), \
        "Style, content and input images should be the same size"


    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.CenterCrop(imsize),  # crop if image is no a square
    transforms.ToTensor()])  # transform it into a torch tensor


unloader = transforms.ToPILImage()  # reconvert into PIL image


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, title=None):
    plt.figure()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
