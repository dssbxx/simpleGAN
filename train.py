import torch
from dataset import AnimeData
from torch.utils.data import DataLoader
from torch import nn,optim
from torch.autograd import Variable
from generator import Generator 
from discriminator import Discriminator 
from torchvision import transforms
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math
import argparse
import time

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--tags', required=True, help='path to tags')
	parser.add_argument('--imgs', required=True, help='path to images')
	parser.add_argument('--g', default='checkpoints/gen', help='path to save the generator trained model')
	parser.add_argument('--d', default='checkpoints/dis', help='path to save the discriminator trained model')
	parser.add_argument('--bs', type=int, default=64, help='number of batch size for load dataset')
	parser.add_argument('--epochs', type=int, default=30, help='number of epochs for train')
	parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate for generator')
	parser.add_argument('--lr_d', type=float, default=0.00005, help='learning rate for discriminator')
	parser.add_argument('--betas', type=float, default=(0.5, 0.999), help='betas for adam optimizer')
	parser.add_argument('--check', action='store_true', help='check the gradient for each step to avoid nan/inf problem')
	parser.add_argument('--noise', type=int, default=100, help='number of noise dimension')
	parser.add_argument('--momentum', type=float, default=0.8, help='number of momentum for batchnormalization')
	parser.add_argument('--label_smoothing', action='store_true', help='do label smoothing for labels')

	return parser

def main(args):

	#transformer
	transform = transforms.Compose([
	transforms.Resize(64),
	transforms.ToTensor(),
	transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])
	])

	#dateset
	anime = AnimeData(args.tags, args.imgs, transform=transform)
	dataloder = DataLoader(anime, batch_size=args.bs, shuffle=True) 

	#model
	gen = Generator(args.noise, momentum=args.momentum)
	dis = Discriminator(momentum=args.momentum)

	#criterion
	criterion = nn.BCELoss()

	if torch.cuda.is_available():
		gen = gen.cuda()
		dis = dis.cuda()
		criterion = criterion.cuda()

	#optimizer
	optimizer_gen = optim.Adam(gen.parameters(), lr=args.lr_g, betas=args.betas)
	optimizer_dis = optim.Adam(dis.parameters(), lr=args.lr_d, betas=args.betas)

	loss_history_d = []
	loss_history_g = []
	out_history_true_d = []
	out_history_fake_d = []
	out_history_fake_g = []

	for epoch in range(args.epochs):
		print('----------------start epoch %d ---------------' % epoch)
		step = 0
		for data in dataloder:
			step += 1
			start = time.time()
			img = Variable(data)
			noise = Variable(torch.randn(img.shape[0], args.noise))
			labels_true = Variable(torch.ones(img.shape[0], 1))
			labels_fake = Variable(torch.zeros(img.shape[0], 1))
			if args.label_smoothing:
				labels_true = labels_true - torch.rand(img.shape[0], 1) * 0.1
				labels_fake = labels_fake + torch.rand(img.shape[0], 1) * 0.1

			#train on GPU
			if torch.cuda.is_available():
				img = img.cuda()
				noise = noise.cuda()
				labels_true = labels_true.cuda()
				labels_fake = labels_fake.cuda()

			#train D
			out_true_d = dis(img)
			out_fake_d = dis(gen(noise))
			out_history_true_d.append(torch.mean(out_true_d).item())
			out_history_fake_d.append(torch.mean(out_fake_d).item())
			#d_loss_ture = -torch.mean(labels_true * torch.log(out_true_d) + (1. - labels_true) * torch.log(1. - out_true_d))
			loss_true_d = criterion(out_true_d, labels_true)
			loss_fake_d = criterion(out_fake_d, labels_fake)
			loss_d = loss_true_d + loss_fake_d
			optimizer_dis.zero_grad()
			loss_d.backward()
		
			if args.check:
				print('>>>>>>>>>>check_d_grad<<<<<<<<<<')
				try:
					check_grad(dis, 'conv2.weight')
				except ValueError as e:
					print(e)
					show(loss_history_d, loss_history_g, out_history_true_d, out_history_fake_d, out_history_fake_g)
					torch.save(dis.state_dict(), os.path.join(os.getcwd(), args.d, 'bad.pth'))
					torch.save(gen.state_dict(), os.path.join(os.getcwd(), args.g, 'bad.pth'))
					return 
			loss_history_d.append(loss_d.item())
			optimizer_dis.step()
			
			#train G
			noise = Variable(torch.randn(img.shape[0], args.noise))
			if torch.cuda.is_available():
				noise = noise.cuda()
			out_fake_g = dis(gen(noise))
			labels_fake = 1. - labels_fake
			out_history_fake_g.append(torch.mean(out_fake_g).item())
			loss_g = criterion(out_fake_g, labels_fake)
			optimizer_gen.zero_grad()
			loss_g.backward()

			if args.check:
				print('>>>>>>>>>>check_g_grad<<<<<<<<<<')
				try:
					check_grad(gen, 'convTrans.weight')
				except ValueError as e:
					print(e)
					show(loss_history_d, loss_history_g, out_history_true_d, out_history_fake_d, out_history_fake_g)
					torch.save(dis.state_dict(), os.path.join(os.getcwd(), args.d, 'bad.pth'))
					torch.save(gen.state_dict(), os.path.join(os.getcwd(), args.g, 'bad.pth'))
					return 
			loss_history_g.append(loss_g.item())
			optimizer_gen.step()
			end = time.time()
			print('epoch: %d  step: %d  d_true: %.2f  d_fake: %.2f  g_fake: %.2f time: %.2f' % (epoch, step, out_history_true_d[-1], out_history_fake_d[-1], out_history_fake_g[-1], end-start))

		#save model
		torch.save(dis.state_dict(), os.path.join(os.getcwd(), args.d, '{}.pth'.format(epoch)))
		torch.save(gen.state_dict(), os.path.join(os.getcwd(), args.g, '{}.pth'.format(epoch)))

def check_grad(model, target=None):
	for param in model.named_parameters():
		name, weight = param
		if name == target:
			print(name)
			print(weight.grad)
			if any(map(math.isinf, weight.grad.view(-1,1))) or any(map(math.isnan, weight.grad.view(-1,1))):
				raise ValueError('Get an inf/nan problem!')
			break
	print('\n')

def show(ld, lg, td, fd, fg):
	plt.subplot(3,2,1)
	plt.plot(ld)
	plt.subplot(3,2,2)
	plt.plot(lg)
	plt.subplot(3,2,3)
	plt.plot(td)
	plt.subplot(3,2,4)
	plt.plot(fd)
	plt.subplot(3,2,5)
	plt.plot(fg)
	plot.savefig('out.jpg')

if __name__ == '__main__':
	parser = get_parser()
	args = parser.parse_args()
	main(args)
