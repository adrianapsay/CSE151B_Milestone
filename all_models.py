##### Baseline model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(identity)
        return self.relu(out)

class SimpleCNN(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, kernel_size=3, init_dim=64, depth=4, dropout_rate=0.2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim
        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim))
            if i < depth - 1:
                current_dim *= 2
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final = nn.Sequential(
            nn.Conv2d(current_dim, current_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(current_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim // 2, n_output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.initial(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.final(self.dropout(x))
        
### DeepCNN
class DeepCNN(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, kernel_size=3, init_dim=64, depth=6, dropout_rate=0.2):
        super().__init__()

        # Initial conv layer
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )

        # A stack of 'depth' convolutional blocks
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(init_dim, init_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(init_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(init_dim, init_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(init_dim),
                nn.ReLU(inplace=True),
            ))

        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate)

        # Final projection to the two output channels (tas, pr)
        self.final = nn.Sequential(
            nn.Conv2d(init_dim, init_dim // 2, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_dim // 2, n_output_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.initial(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        return self.final(x)

### ConvLSTMNet
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, x):
        B, T, C, H, W = x.size()
        h = [torch.zeros(B, layer.conv.out_channels // 4, H, W, device=x.device) for layer in self.layers]
        c = [torch.zeros(B, layer.conv.out_channels // 4, H, W, device=x.device) for layer in self.layers]

        skip_connections = []

        for t in range(T):
            input_ = x[:, t]
            for i, layer in enumerate(self.layers):
                h[i], c[i] = layer(input_, h[i], c[i])
                input_ = h[i]
            if t == T - 1:
                skip_connections.append(input_)  # final hidden state for skip

        return h[-1], skip_connections  # final output and skip connection


class UNetDecoderBlock(nn.Module):
    def __init__(self, up_channels, skip_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Dropout2d(dropout_rate),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)



class ConvLSTMUNet(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, T, kernel_size=3, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.T = T
        self.C = n_input_channels // T

        self.convlstm = ConvLSTM(self.C, hidden_dim, kernel_size, num_layers=2)

        self.dec1 = UNetDecoderBlock(
            up_channels=hidden_dim,
            skip_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            dropout_rate=dropout_rate
        )

        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, n_output_channels, kernel_size=1)
        )


    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.T, self.C, H, W)
        x, skips = self.convlstm(x)
        x = self.dec1(x, skips[-1])
        x = self.final(x)
        x = F.interpolate(x, size=(48, 72), mode="bilinear", align_corners=False)
        return x

#### UNet CNN (v1)
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetCNN(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, init_dim=64, dropout_rate=0.2):
        super().__init__()

        # Encoder
        self.enc1 = UNetBlock(n_input_channels, init_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(init_dim, init_dim * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(init_dim * 2, init_dim * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(init_dim * 4, init_dim * 2, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(init_dim * 4, init_dim * 2)
        self.up1 = nn.ConvTranspose2d(init_dim * 2, init_dim, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(init_dim * 2, init_dim)

        # Final output
        self.final = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(init_dim, n_output_channels, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)                      # [B, 64, 48, 72]
        enc2 = self.enc2(self.pool1(enc1))       # [B, 128, 24, 36]
        bottleneck = self.bottleneck(self.pool2(enc2))  # [B, 256, 12, 18]

        # Decoder
        dec2 = self.up2(bottleneck)              # [B, 128, 24, 36]
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)                    # [B, 64, 48, 72]
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return self.final(dec1)                  # [B, 2, 48, 72]

### UNet (v2) - dual heads
##### UNet CNN
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetCNN(nn.Module):
    def __init__(self, n_input_channels, init_dim=64, dropout_rate=0.2):
        super().__init__()

        # Encoder
        self.enc1 = UNetBlock(n_input_channels, init_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(init_dim, init_dim * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(init_dim * 2, init_dim * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(init_dim * 4, init_dim * 2, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(init_dim * 4, init_dim * 2)
        self.up1 = nn.ConvTranspose2d(init_dim * 2, init_dim, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(init_dim * 2, init_dim)

        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate)

        # Dual output heads: one for tas, one for pr
        self.tas_head = nn.Conv2d(init_dim, 1, kernel_size=1)
        self.pr_head  = nn.Conv2d(init_dim, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        # Decoder
        dec2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        dec1 = self.dropout(dec1)

        # Forward through both heads
        tas_out = self.tas_head(dec1)
        pr_out  = self.pr_head(dec1)

        return torch.cat([tas_out, pr_out], dim=1)  # Output: [B, 2, 48, 72]

### UNetCNN (v3) - with CoordConv
class CoordConv(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        yy = torch.linspace(-1, 1, H, device=x.device).view(1,1,H,1).expand(B,1,H,W)
        xx = torch.linspace(-1, 1, W, device=x.device).view(1,1,1,W).expand(B,1,H,W)
        return torch.cat([x, xx, yy], dim=1)

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=k//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNetCNN(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, init_dim=64, dropout_rate=0.2):
        super().__init__()
        # Coord embedding + first encoder
        self.coord  = CoordConv()
        self.enc1   = UNetBlock(n_input_channels + 2, init_dim)
        self.pool1  = nn.MaxPool2d(2)
        # Second encoder
        self.enc2   = UNetBlock(init_dim, init_dim*2)
        self.pool2  = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = UNetBlock(init_dim*2, init_dim*4)
        # Decoder
        self.up2    = nn.ConvTranspose2d(init_dim*4, init_dim*2, 2, stride=2)
        self.dec2   = UNetBlock(init_dim*4, init_dim*2)
        self.up1    = nn.ConvTranspose2d(init_dim*2, init_dim,   2, stride=2)
        self.dec1   = UNetBlock(init_dim*2, init_dim)
        # Dropout + final projection
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final   = nn.Conv2d(init_dim, n_output_channels, kernel_size=1)

    def forward(self, x):
        x  = self.coord(x)         # add coordinate channels
        e1 = self.enc1(x)          # [B, init_dim, 48,72]
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.dropout(d1)
        return self.final(out)     # [B, n_output_channels, 48,72]

### UNetCNN (builds upon v3) - with pr log transform
class CoordConv(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        yy = torch.linspace(-1, 1, H, device=x.device).view(1,1,H,1).expand(B,1,H,W)
        xx = torch.linspace(-1, 1, W, device=x.device).view(1,1,1,W).expand(B,1,H,W)
        return torch.cat([x, xx, yy], dim=1)

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=k//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNetCNN(nn.Module):
    def __init__(self, n_input_channels, init_dim=64, dropout_rate=0.2):
        super().__init__()
        self.coord = CoordConv()
        self.enc1 = UNetBlock(n_input_channels + 2, init_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(init_dim, init_dim * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(init_dim * 2, init_dim * 4)

        # shared upsampling blocks
        self.up2 = nn.ConvTranspose2d(init_dim * 4, init_dim * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(init_dim * 2, init_dim, 2, stride=2)

        # separate decoders for tas and pr
        self.dec2_tas = UNetBlock(init_dim * 4, init_dim * 2)
        self.dec1_tas = UNetBlock(init_dim * 2, init_dim)
        self.final_tas = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(init_dim, 1, kernel_size=1)
        )

        self.dec2_pr = UNetBlock(init_dim * 4, init_dim * 2)
        self.dec1_pr = UNetBlock(init_dim * 2, init_dim)
        self.final_pr = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(init_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.coord(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))
        
        up2 = self.up2(b)
    
        # tas path
        d2_tas = self.dec2_tas(torch.cat([up2, e2], dim=1))
        up1_tas = self.up1(d2_tas)
        d1_tas = self.dec1_tas(torch.cat([up1_tas, e1], dim=1))
        out_tas = self.final_tas(d1_tas)
    
        # pr path
        d2_pr = self.dec2_pr(torch.cat([up2, e2], dim=1))
        up1_pr = self.up1(d2_pr)
        d1_pr = self.dec1_pr(torch.cat([up1_pr, e1], dim=1))
        out_pr = self.final_pr(d1_pr)
    
        return torch.cat([out_tas, out_pr], dim=1)  # shape: [B, 2, H, W]