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


class mimo_UNet(nn.Module):
    def __init__(self, in_point, h, w, n_branch, bilinear=False):
        super(mimo_UNet, self).__init__()
        self.in_point = in_point
        self.h = h
        self.w = w
        self.n_branch = n_branch
        self.bilinear = bilinear

        self._init_first_layer(n_branch, in_point, h * w)
        self.inc = DoubleConv(n_branch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self._init_final_layer(n_branch)

    def _init_first_layer(self, n_branch, in_point, out_point):
        first_layer = []
        for _ in range(n_branch):
            first_layer.append(nn.Linear(in_point, out_point))
        self.first_layer = nn.ModuleList(first_layer)

    def _init_final_layer(self, n_branch):
        final_layer = []
        for _ in range(n_branch):
            final_layer.append(OutConv(64, 1))
        self.final_layer = nn.ModuleList(final_layer)

    def forward_first(self, x, n_branch):
        B = x.shape[0]
        y = []
        for i in range(n_branch):
            if x.size(1) == 1:
                x_member = x
            else:
                x_member = x[:, i:(i + 1)]
            y.append(self.first_layer[i](x_member))
        y_cat = torch.cat([y[0].reshape(B, 1, self.h, self.w), y[1].reshape(B, 1, self.h, self.w)], dim=1)
        return y_cat

    def forward_final(self, x, n_branch):
        B = x.shape[0]
        out = {}
        list_y = []
        for i in range(n_branch):
            y_out = self.final_layer[i](x).reshape(B, self.h * self.w)
            out["final_" + str(i)] = y_out
            list_y.append(y_out)
        out["final"] = torch.stack(list_y, dim=0).mean(dim=0)
        return out

    def forward(self, x):
        x = self.forward_first(x, self.n_branch)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print("x5.shape", x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.forward_final(x, self.n_branch)
        return logits


def encoder_UNet(n_channels, f_maps, bilinear=False):
    inc = DoubleConv(n_channels, f_maps)
    down1 = Down(f_maps, f_maps * 2)
    down2 = Down(f_maps * 2, f_maps * 4)
    down3 = Down(f_maps * 4, f_maps * 8)
    factor = 2 if bilinear else 1
    down4 = Down(f_maps * 8, (f_maps * 16) // factor)

    return nn.ModuleList([inc, down1, down2, down3, down4])


class mv_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_branch=2, bilinear=False):
        super(mv_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_branch = n_branch
        self.n_classes = n_classes
        self.bilinear = bilinear
        f_maps = 32
        # if cat, f_maps = f_maps // 2
        self.encoder1 = encoder_UNet(n_channels, f_maps // 2)
        self.encoder2 = encoder_UNet(n_channels, f_maps // 2)
        factor = 2 if bilinear else 1
        self.up1 = Up(f_maps * 16, (f_maps * 8) // factor, bilinear)
        self.up2 = Up(f_maps * 8, (f_maps * 4) // factor, bilinear)
        self.up3 = Up(f_maps * 4, (f_maps * 2) // factor, bilinear)
        self.up4 = Up(f_maps * 2, f_maps, bilinear)
        self.outc = OutConv(f_maps, n_classes)

    def forward(self, x):
        x_1 = x[:, 0:3]
        x_2 = x[:, 3:6]
        x_list1, x_list2 = [], []
        for encoder in self.encoder1:
            x_1 = encoder(x_1)
            x_list1.append(x_1)
        for encoder in self.encoder2:
            x_2 = encoder(x_2)
            x_list2.append(x_2)

        x0_1, x0_2, x0_3, x0_4, x0_5 = x_list1[0], x_list1[1], x_list1[2], x_list1[3], x_list1[4]
        x1_1, x1_2, x1_3, x1_4, x1_5 = x_list2[0], x_list2[1], x_list2[2], x_list2[3], x_list2[4]
        ###############
        # cat到一起
        x1 = torch.cat([x0_1, x1_1], dim=1)
        x2 = torch.cat([x0_2, x1_2], dim=1)
        x3 = torch.cat([x0_3, x1_3], dim=1)
        x4 = torch.cat([x0_4, x1_4], dim=1)
        x5 = torch.cat([x0_5, x1_5], dim=1)
        # 加到一起
        # x1 = x0_1 + x1_1
        # x2 = x0_2 + x1_2
        # x3 = x0_3 + x1_3
        # x4 = x0_4 + x1_4
        # x5 = x0_5 + x1_5
        # 学习如何加在一起

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class MT_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(MT_UNet, self).__init__()
        # student network
        self.sub_net1 = UNet(n_channels, n_classes, bilinear)
        # teacher network
        self.sub_net2 = UNet(n_channels, n_classes, bilinear)
        # detach the teacher model
        for param in self.sub_net2.parameters():
            param.detach_()

    def forward(self, x, step, cur_iter=1):
        if not self.training:
            s_out = self.sub_net1(x)
            return s_out

        # copy the parameters from teacher to student
        if cur_iter == 1:
            for t_param, s_param in zip(self.sub_net2.parameters(), self.sub_net1.parameters()):
                t_param.data.copy_(s_param.data)

        s_out = self.sub_net1(x)
        with torch.no_grad():
            t_out = self.sub_net2(x)
        if step == 1:
            self._update_ema_variables(ema_decay=0.99)

        return s_out, t_out

    def _update_ema_variables(self, ema_decay=0.99):
        for t_param, s_param in zip(self.sub_net2.parameters(), self.sub_net1.parameters()):
            t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1 - ema_decay)


class Twin_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Twin_UNet, self).__init__()

        self.sub_net1 = UNet(n_channels, n_classes, bilinear)
        self.sub_net2 = UNet(n_channels, n_classes, bilinear)
        # self.sub_net2 = mv_UNet(n_channels, n_classes, bilinear)
        # self.sub_net3 = mv_UNet(n_channels, n_classes, bilinear)

    def forward(self, x, step):
        if step == 1:
            out_1 = self.sub_net1(x)
            return out_1
        elif step == 2:
            out_2 = self.sub_net2(x)
            return out_2
        # elif step == 3:
        #     out_diff = self.sub_net3(x)
        #     return out_diff


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


class ZNet(nn.Module):
    def __init__(self, in_size, h, w, c):
        super(ZNet, self).__init__()
        self.h = h
        self.w = w
        self.infc = nn.Sequential(
            nn.Linear(in_size, h * w // (16 * 16 * 2)),
            nn.GELU(),
            nn.Linear(h * w // (16 * 16 * 2), (h // 16) * (w // 16)))

        self.down1 = ResConv(c, 64)
        self.down2 = ResConv(64, 256)
        # self.down3 = ResConv(128, 256)
        # self.down4 = ResConv(256, 512)
        self.up1 = Up_samp(64, 256, 128)  # in 16 12
        self.up2 = Up_samp(64, 256, 128)  # in 32 25
        self.up3 = Up_samp(64, 256, 128)  # in 64 50
        self.up4 = Up_samp(64, 64)  # in 128 100
        self.outc = OutConv(64, c)

    def forward(self, x):
        B, N, _ = x.shape
        x1 = self.infc(x).reshape(B, N, self.h // 16, self.w // 16)

        x = self.down1(x1)
        x = self.down2(x)
        x = self.up1(x)
        x = F.interpolate(x, [25, 25], mode='bilinear', align_corners=True)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits


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
