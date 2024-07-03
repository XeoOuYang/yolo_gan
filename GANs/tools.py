import os.path

import torch

from GANs.models.mlp_gan import show_tensor_image


def check_dir(path):
    if path == '': return
    if os.path.isdir(path) and not os.path.exists(path):
        os.makedirs(path)
    else:
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def save_all(model, path):
    """保存整个网络，可以继续训练"""
    torch.save(model, path)

def load_all(path):
    return torch.load(path)

def save_state_dict(model, path):
    """只保留权重，用于运行分享"""
    torch.save(model.state_dict(), path)

def load_state_dic(model, path):
    model.load_state_dict(torch.load(path))

def get_noise(n_sample, z_dim, device='cpu'):
    return torch.randn(n_sample, z_dim, device=device)

def get_discriminator_loss(gen, disc, criterion, real, batch_size, z_dim, device):
    fake_noise = get_noise(batch_size, z_dim, device)
    fake = gen(fake_noise)  # [batch_size, im_dim]
    disc_fake_predict = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_predict, torch.zeros_like(disc_fake_predict))
    disc_real_predict = disc(real)
    disc_real_loss = criterion(disc_real_predict, torch.ones_like(disc_real_predict))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss

def get_generator_loss(gen, disc, criterion, batch_size, z_dim, device):
    fake_noise = get_noise(batch_size, z_dim, device)
    fake = gen(fake_noise)
    disc_fake_predict = disc(fake)
    disc_fake_loss = criterion(disc_fake_predict, torch.ones_like(disc_fake_predict))
    return disc_fake_loss

if __name__ == '__main__':
    batch_size= 16
    tensors = torch.ones(batch_size, 1, 28, 28)
    show_tensor_image(tensors, batch_size, (1, 28, 28))