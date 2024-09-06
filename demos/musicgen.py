# from einops import rearrange,repeat
# import torch
# import numpy as np
#
# image1 = torch.zeros(220, 2, 224)
# #im2 = rearrange(image1, "n d -> n () d")
# im2 = rearrange(image1, "... d -> (...) d")
# print(im2.shape)
# image4 = torch.tensor(np.array([1, 2, 3, 4]))
# im3 = repeat(image4, "n -> n d", d=3)
# print(im3.shape)
#
# from mindspore import ops, Tensor
#
# input_tensor = ops.zeros((220, 2, 224))
# print(input_tensor.shape)
# output = input_tensor.reshape(-1, input_tensor.shape[-1])
# # output = ops.unsqueeze(input_tensor, dim=1)
# # input_tensor2 = Tensor(np.array([1, 2, 3, 4]))
# # output2 = input_tensor2.repeat(3).reshape(-1, 3)
# # print(output2.shape)

from audiocraft.models import MusicGen
#from audiocraft.models import MultiBandDiffusion
import torch
USE_DIFFUSION_DECODER = False
# Using small model, better results would be obtained with `medium` or `large`.
model = MusicGen.get_pretrained('/home/litingyu/08c/small')
# if USE_DIFFUSION_DECODER:
#     mbd = MultiBandDiffusion.get_mbd_musicgen()

from audiocraft.audiocraft_utils.notebook import display_audio

output = model.generate(
    descriptions=[
        'funky guitar， pop songs, EXO'
        #'80s pop track with bassy drums and synth',
        #'90s rock song with loud guitars and heavy drums',
        #'Progressive rock drum and bass solo',
        #'Punk Rock song with loud drum and power guitar',
        #'Bluesy guitar instrumental with soulful licks and a driving rhythm section',
        #'Jazz Funk song with slap bass and powerful saxophone',
        #'drum and bass beat with intense percussions'
    ],
    progress=True, return_tokens=True
)
#display_audio(output[0], sample_rate=32000)
import scipy

sampling_rate = 44100
scipy.io.wavfile.write("testexo.wav", rate=sampling_rate, data=output[0].numpy())

# import mindspore as ms
# from mindspore import nn, ops

# class GradNetWrtX(nn.Cell):
#     def __init__(self, net):
#         super(GradNetWrtX, self).__init__()
#         self.net = net
#
#     @property
#     def get_something(self):
#         return 100
#
#     @property
#     def get_twothing(self):
#         a = self.get_something
#         b = a+1
#         return b
#     def construct(self, x, y):
#
#         gradient_function = ms.grad(self.net)
#         return gradient_function(x, y)
#
# test = GradNetWrtX(nn.Dense(10, 40))
# b = test.get_twothing
# print(b)
