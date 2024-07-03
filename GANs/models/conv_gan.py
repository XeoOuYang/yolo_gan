from torch import nn
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt

def show_tensor_image(image_tensor, batch_size, save_path, show_grid=False):
    image_tensor = (image_tensor + 1) / 2
    images = image_tensor.detach().cpu()
    image_grid = make_grid(images[:batch_size], nrow=5)
    # 保存结果
    save_image(image_grid, save_path)
    # 显示结果
    if show_grid:
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()

def get_generator_block(input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
    if not final_layer:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            nn.Tanh()
        )

class Generator(nn.Module):
    def __init__(self, z_dim=10, im_channels=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            get_generator_block(hidden_dim * 2, hidden_dim),
            get_generator_block(hidden_dim, im_channels, kernel_size=4, stride=2, final_layer=True)
        )

    def forward(self, noise):
        noise = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(noise)

def get_discriminator_block(input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
    if not final_layer:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride)
        )

class Discriminator(nn.Module):
    def __init__(self, im_channels=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_channels, hidden_dim),
            get_discriminator_block(hidden_dim, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, 1, final_layer=True)
        )

    def forward(self, image):
        image = self.disc(image)
        return image.view(len(image), -1)
