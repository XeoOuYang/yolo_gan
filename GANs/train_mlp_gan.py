import os.path

import torch.cuda
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from GANs.tools import get_discriminator_loss, get_generator_loss, get_noise, check_dir, save_all, load_all

cwd = os.getcwd()
runs = run_epoch_path = os.path.join(cwd, 'runs', 'mlp')
check_dir(runs)
outs = os.path.join(cwd, 'outs', 'mlp')
check_dir(outs)

def run_mpl_gan_train():
    from GANs.models.mlp_gan import Generator, Discriminator, show_tensor_image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 数据加载
    batch_size = 128
    dataloader = DataLoader(
        MNIST(cwd, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True
    )
    # loss、gen、gen_opt、disc、disc_opt
    noise_dim = 64      # 一张图片的最终表征
    lr = 1e-5
    criterion = nn.BCEWithLogitsLoss()
    gen = Generator(noise_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
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
            real = real.view(run_batch_size, -1).to(device)
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
        print(f'Epoch {epoch+1}/{n_epochs}: mlp generator loss: {mean_gen_loss}, discriminator loss: {mean_disc_loss}')
        mean_disc_loss = 0
        mean_gen_loss = 0
        # 测试结果
        valid_batch_size = 25
        result_epoch_path = os.path.join(runs, f'mlp_{epoch}.jpg')
        fake_noise = get_noise(valid_batch_size, noise_dim, device=device)
        gen.eval()
        fake = gen(fake_noise)
        show_tensor_image(fake, valid_batch_size, result_epoch_path)
    # 保存模型结果
    save_all(gen, os.path.join(outs, 'G.pth'))
    save_all(disc, os.path.join(outs, 'D.pth'))

def run_mlp_gan():
    from GANs.models.mlp_gan import show_tensor_image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = load_all(os.path.join(outs, 'G.pth')).to(device)
    gen.eval()
    # 测试结果
    noise_dim = 64
    valid_batch_size = 25
    result_epoch_path = os.path.join(runs, f'mlp_gen.jpg')
    fake_noise = get_noise(valid_batch_size, noise_dim, device=device)
    fake = gen(fake_noise)
    show_tensor_image(fake, valid_batch_size, result_epoch_path, show_grid=True)

if __name__ == '__main__':
    # 多层感知网络gann
    run_mpl_gan_train()
    # 运行模型
    run_mlp_gan()