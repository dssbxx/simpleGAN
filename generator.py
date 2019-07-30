import torch
import torch.nn as nn
from torch.autograd import Variable

class Upsample(nn.Module):
	"""docstring for Bottleneck"""
	def __init__(self, in_channels, out_chennels, kernel_size, stride, padding, momentum):
		super(Upsample, self).__init__()
		self.bm = nn.BatchNorm2d(out_chennels, momentum)
		self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
		self.conv_trans = nn.ConvTranspose2d(in_channels, out_chennels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		
		self.conv_trans.weight.data.normal_(0.0, 0.02)
		self.bm.weight.data.normal_(0.02, 1.0)
		self.bm.bias.data.fill_(0)

	def forward(self, x):
		x = self.conv_trans(x)
		x = self.bm(x)
		x = self.leakyrelu(x)
		
		return x
		
class ExtraLayer(nn.Module):
	def __init__(self, in_channels, out_chennels, kernel_size, stride, padding):
		super(ExtraLayer, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_chennels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.bm = nn.BatchNorm2d(out_chennels)
		self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

		self.conv.weight.data.normal_(0.0, 0.02)
		self.bm.weight.data.normal_(1.0, 0.02)
		self.bm.bias.data.fill_(0)

	def forward(self, x):
		x = self.conv(x)
		x = self.bm(x)
		x = self.leakyrelu(x)

		return x

class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self, noise_dim, momentum=0.8):
		super(Generator, self).__init__()
		self.noise_dim = noise_dim

		self.upsample_1 = Upsample(noise_dim, 512, 4, 1, 0, momentum)
		self.upsample_2 = Upsample(512, 256, 4, 2, 1, momentum)
		self.upsample_3 = Upsample(256, 128, 6, 2, 2, momentum)
		self.upsample_4 = Upsample(128, 64, 6, 2, 2, momentum)
		self.conv = ExtraLayer(64, 64, 3, 1, 1)
		self.convTrans = nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2, padding=2, bias=False)
		self.tanh = nn.Tanh()
		
		self.convTrans.weight.data.normal_(0.0, 0.02)

	def forward(self, x):
		x = x.unsqueeze(-1).unsqueeze(-1)
		assert x.shape[1] == self.noise_dim
		x = self.upsample_1(x)
		x = self.upsample_2(x)
		x = self.upsample_3(x)
		x = self.upsample_4(x)
		x = self.conv(x)
		x = self.convTrans(x)
		x = self.tanh(x)
	
		return x
