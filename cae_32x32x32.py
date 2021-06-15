import torch
import torch.nn as nn

def res_block():

    res = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

    return res


class Block(nn.Module):


    def __init__(self):

        super(Block, self).__init__()

        self.block1 = res_block()
        self.block2 = res_block()
        self.block3 = res_block()

    def forward(self, x):
        identity = x

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out += identity
        
        return out


class CAE(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)

    Latent representation: 16x16x16 bits per patch => 30KB per image (for 720p)
    """

    def __init__(self):
        super(CAE, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = Block()

        # 128x32x32
        self.e_block_2 = Block()

        # 128x32x32
        self.e_block_3 = Block()

        
        self.e_conv_3= nn.Sequential(
                        nn.ZeroPad2d((1, 2, 1, 2)),
                        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2))
                         
                    )

        self.e_conv_4= nn.Sequential(
                        nn.ZeroPad2d((1, 1, 1, 1)),
                        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=(2, 2)),
                        nn.Tanh()
                    )


        # DECODER

        # 128x32x32
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        )

        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )

        # 128x32x32
        self.d_block_1 = Block()

        # 128x32x32
        self.d_block_2 = Block()

        # 128x32x32
        self.d_block_3 = Block()

        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        )

        # 3x128x128
        self.d_up_conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(2, 2), stride=(2, 2)),
            nn.Tanh()
        )

    def forward(self, x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
   
        identity = ec2
        eblock1 = self.e_block_1(ec2)
        eblock2 = self.e_block_2(eblock1)
        eblock3 = self.e_block_3(eblock2)
        ec3 = self.e_conv_3(eblock3 + identity)  # in [-1, 1] from tanh activation
        ec3 = self.e_conv_4(ec3)

        # stochastic binarization
        # with torch.no_grad():
        #     out = torch.zeros(ec3.shape)
        #     if torch.cuda.is_available():
        #         out = out.cuda()
        #     out[ec3<0]=0.
        #     out[ec3>=0]=1.

        # encoded tensor
        self.encoded = ec3  # (-1|1) -> (0|1)

        return self.decode(self.encoded)

    def decode(self, encoded):
        #y = encoded * 2.0 - 1  # (0|1) -> (-1|1)
        y = encoded.type(torch.float32)
        uc1 = self.d_up_conv_1(y)
        uc1 = self.d_up_conv_2(uc1)
        identity = uc1
        dblock1 = self.d_block_1(uc1)
        dblock2 = self.d_block_2(dblock1)
        dblock3 = self.d_block_3(dblock2)
        uc2 = self.d_up_conv_3(dblock3 + identity)
        dec = self.d_up_conv_4(uc2)

        return dec
