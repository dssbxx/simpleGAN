import torch
import torch.nn as nn

class Downsample(nn.Module):
	"""docstring for Downsample"""
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, momentum):
		super(Downsample, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.bm = nn.BatchNorm2d(out_channels, momentum)
		self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
		
		self.conv.weight.data.normal_(0.0, 0.02)
		self.bm.weight.data.normal_(1.0, 0.02)
		self.bm.bias.data.fill_(0)

	def forward(self, x):
		x = self.conv(x)
		x = self.bm(x)
		x = self.leaky_relu(x)

		return x

class Discriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, momentum=0.8):
		super(Discriminator, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=2, bias=False)
		self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
		self.downsample_1 = Downsample(64, 128, 6, 2, 2, momentum)
		self.downsample_2 = Downsample(128, 256, 6, 2, 2, momentum)
		self.downsample_3 = Downsample(256, 512, 4, 2, 1, momentum)
		self.conv2 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
		self.sigmoid = nn.Sigmoid()
		
		self.conv1.weight.data.normal_(0.0, 0.02)
		self.conv2.weight.data.normal_(0.0, 0.02)
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.leakyrelu(x)
		x = self.downsample_1(x)
		x = self.downsample_2(x)
		x = self.downsample_3(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)
		x = self.sigmoid(x)

		return x
