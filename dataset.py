import torch
from torch.utils import data
from PIL import Image

'''

tags format:
0:aqua hair,black eyes
1:red hair,green eyes
...

'''

class AnimeData(data.Dataset):
		def __init__(self, tags_path, img_path, transform=None):
			self.tags_list = self._get_tags_list(tags_path)
			self.img_path = img_path
			self.transform = transform

		def _get_tags_list(self, tags_path):
			tags_list = []
			with open(tags_path, 'r') as f:
				for line in f.readlines():
					tags_list.append(line.split(':')[0])

			return tags_list
		
		def __getitem__(self, index):
			data_num = self.tags_list[index]

			img = Image.open(self.img_path+data_num+'.jpg')

			if self.transform is not None:
				img = self.transform(img)

			return img
			
		def __len__(self):
			return len(self.tags_list)
