import torch.nn as nn


class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()

        self.feature = FeatureModule()
        self.pyramid_pooling = PyramidPoolingModule()
        self.decode_feature = DecoderModule()
        self.aux = AuxLossModule()

    def forward(self, x):
        x, x_aux = self.feature(x)
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)

        return output, self.aux(x_aux)
