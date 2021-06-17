import torch


def get_grid(sde, input_channels, input_height, dist, n=4, num_steps=20, transform=None,
             mean=0, std=1, clip=True):
    #TODO right now for sampling we use the
    num_samples = n ** 2
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, input_channels, input_height, input_height).to(sde.T)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)
    sslr = torch.cat([torch.eye(dist.probs.shape[0])[i] for i in list(dist.sample(sample_shape=(num_samples, 1)))]) # [B, sslr_dims(now it is just 10)]

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0, sslr)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)

    if transform is not None:
        y0 = transform(y0)

    if clip:
        y0 = torch.clip(y0, 0, 1)

    # ou_sde.sample(ou_sde.T*50, x)
    y0 = y0.view(
        n, n, input_channels, input_height, input_height).permute(
        2, 0, 3, 1, 4).contiguous().view(input_channels, n * input_height, n * input_height)

    y0 = y0.data.cpu().numpy()
    return y0
