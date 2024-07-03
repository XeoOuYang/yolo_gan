import os.path

import torch.cuda
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from GANs.tools import get_discriminator_loss, get_generator_loss, get_noise, check_dir, save_all, load_all

cwd = os.getcwd()
runs = run_epoch_path = os.path.join(cwd, 'runs', 'conv')
check_dir(runs)
outs = os.path.join(cwd, 'outs', 'conv')
check_dir(outs)

def run_conv_gan_train():
    from GANs.models.conv_gan import Generator, Discriminator, show_tensor_image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 数据加载
    batch_size = 128
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), 0.5)
    ])
    dataloader = DataLoader(
        MNIST(cwd, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True
    )
    # loss、gen、gen_opt、disc、disc_opt
    noise_dim = 64      # 一张图片的最终表征
    lr = 2e-4
    beta_1 = 0.5
    beta_2 = 0.999
    criterion = nn.BCEWithLogitsLoss()
    gen = Generator(noise_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
    def init_weights(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
        if isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
            nn.init.constant_(module.bias, 0)
    gen = gen.apply(init_weights)
    disc = disc.apply(init_weights)
    # 开始训练
    n_epochs = 100
    mean_gen_loss = 0
    mean_disc_loss = 0
    for epoch in range(n_epochs):
        gen.train()
        disc.train()
        total_size = len(dataloader)
        for real, _ in tqdm.tqdm(dataloader):
            run_batch_size = len(real)
            real = real.to(device)
            # 训练生成网络
            gen_opt.zero_grad()
            gen_loss = get_generator_loss(gen, disc, criterion, run_batch_size, noise_dim, device)
            gen_loss.backward()
            gen_opt.step()
            mean_gen_loss += gen_loss.item()/total_size
            # 训练判别网络
            disc_opt.zero_grad()
            disc_loss = get_discriminator_loss(gen, disc, criterion, real, run_batch_size, noise_dim, device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()
            mean_disc_loss += disc_loss.item()/total_size
        # 打印本次epoch数据
        print()
        print(f'Epoch {epoch+1}/{n_epochs}: conv generator loss: {mean_gen_loss}, discriminator loss: {mean_disc_loss}')
        mean_disc_loss = 0
        mean_gen_loss = 0
        # 测试结果
        valid_batch_size = 25
        result_epoch_path = os.path.join(runs, f'conv_{epoch}.jpg')
        fake_noise = get_noise(valid_batch_size, noise_dim, device=device)
        gen.eval()
        fake = gen(fake_noise)
        show_tensor_image(fake, valid_batch_size, result_epoch_path)
    # 保存模型结果
    save_all(gen, os.path.join(outs, 'G.pth'))
    save_all(disc, os.path.join(outs, 'D.pth'))

def run_conv_gan():
    from GANs.models.conv_gan import show_tensor_image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = load_all(os.path.join(outs, 'G.pth')).to(device)
    gen.eval()
    # 测试结果
    noise_dim = 64
    valid_batch_size = 25
    result_epoch_path = os.path.join(runs, f'conv_gen.jpg')
    fake_noise = get_noise(valid_batch_size, noise_dim, device=device)
    fake = gen(fake_noise)
    show_tensor_image(fake, valid_batch_size, result_epoch_path, show_grid=True)


if __name__ == '__main__':
    # 卷积神经网络gan
    run_conv_gan_train()
    # 运行模型
    run_conv_gan()