import torch.nn as nn

# Pretrained VGG19 feature extractor to compute content loss.
# Content loss is the L2 loss between the features of the rendered image and the content image.

# Weights available at the following links:
### -  https://www.dropbox.com/s/hv7b4eajrj7isyq/vgg_weights.pickle?dl=1
### -  https://github.com/ftokarev/tf-vgg-weights/raw/master/vgg19_weights_normalized.h5
class VGG19(nn.Module):
    def __init__(self, weights_path):
        super(VGG19, self).__init__()
