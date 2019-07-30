import torch
from torchvision.utils import save_image
from generator import Generator
from torch.autograd import Variable

NOISE_DIM = 100
G_LOAD_PATH = r'/home/zhoutao/repo/SimpleGAN/checkpoints/gen/15.pth'

SAVE_PATH = r'fake_15.jpg'


def save_fake(fake_img, path):
	img = 0.5  * (fake_img + 1)
	img = img.clamp(0, 255).view(-1, 3, 64, 64)
	save_image(img, path)

if __name__ == '__main__':
	gen = Generator(NOISE_DIM, 0.8)
	gen.load_state_dict(torch.load(G_LOAD_PATH))
	gen.cuda()

	noise = Variable(torch.randn(32,NOISE_DIM)).cuda()

	fake_img = gen(noise)
	save_fake(fake_img, SAVE_PATH)
	print('save fake images in {}'.format(SAVE_PATH))
