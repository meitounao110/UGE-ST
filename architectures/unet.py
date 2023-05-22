from architectures.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        f_maps = 64

        self.inc = DoubleConv(n_channels, f_maps)
        self.down1 = Down(f_maps, f_maps * 2)
        self.down2 = Down(f_maps * 2, f_maps * 4)
        self.down3 = Down(f_maps * 4, f_maps * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(f_maps * 8, (f_maps * 16) // factor)
        self.up1 = Up(f_maps * 16, (f_maps * 8) // factor, bilinear)
        self.up2 = Up(f_maps * 8, (f_maps * 4) // factor, bilinear)
        self.up3 = Up(f_maps * 4, (f_maps * 2) // factor, bilinear)
        self.up4 = Up(f_maps * 2, f_maps, bilinear)
        self.outc = OutConv(f_maps, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class multi_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(multi_UNet, self).__init__()
        self.sub_net1 = UNet(n_channels, n_classes, bilinear)
        self.sub_net2 = UNet(n_channels, n_classes, bilinear)
        self.sub_net3 = UNet(n_channels, n_classes, bilinear)

    def forward(self, x):
        y1 = self.sub_net1(x)
        y2 = self.sub_net2(x)
        y3 = self.sub_net3(x)
        return y1, y2, y3


if __name__ == '__main__':
    from itertools import permutations

    x = torch.randn(4, 6, 200, 200)  # .cuda()
    z = torch.randn(4, 3, 200, 200).cuda()
    p = []
    for i in range(x.shape[0]):
        p.append([x[i, ...], z[i, ...]])
    p = list(permutations(p, 2))
    p_diff = []
    for i in range(len(p)):
        p_diff.append(p[i][0][1] - p[i][1][1])
        p[i] = torch.cat([p[i][0][0], p[i][1][0]], dim=0)
    p = torch.stack(p, dim=0)
    p_diff = torch.stack(p_diff, dim=0)
    # x = torch.randn(4, 3, 50, 50)
    # x = torch.stack([x1, x2], dim=1)
    print("x.shape", x.shape)
    model = Twin_UNet(3, 3)
    # torch.save(model, "/mnt/zyy/reconstruction/model.pth")
    # net_path = "/mnt/zyy/reconstruction/model.pth"
    # checkpoint = torch.load(net_path)
    print(model)
    # y = model(x)
    y = model(x, step=2)
    print(y["final"].shape)
