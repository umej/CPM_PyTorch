from preprocess.gen_data import LSP_DATA
from preprocess.Transformers import Compose, RandomCrop, RandomResized, TestResized
from utils import AverageMeter
import cpm
import torch.utils.data.dataloader
import torch.nn as nn

criterion = nn.MSELoss().cuda()

model = cpm.CPM(k=14).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_losses = AverageMeter()
val_losses = AverageMeter()
min_losses = 999

epoch = 0
while epoch < 20:
	print('epoch ', epoch)
	"""--------Train--------"""
	# Training data
	data = LSP_DATA('lspet', 'F:/Python/PyCharmWorkspace/CPM/lspet/', 8, Compose([RandomResized(), RandomCrop(368)]))
	train_loader = torch.utils.data.dataloader.DataLoader(data, batch_size=8)
	for j, data in enumerate(train_loader):
		inputs, heatmap, centermap = data

		inputs = inputs.cuda()
		heatmap = heatmap.cuda(async=True)
		centermap = centermap.cuda(async=True)

		input_var = torch.autograd.Variable(inputs)
		heatmap_var = torch.autograd.Variable(heatmap)
		centermap_var = torch.autograd.Variable(centermap)

		heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, centermap_var)

		loss1 = criterion(heat1, heatmap_var)
		loss2 = criterion(heat2, heatmap_var)
		loss3 = criterion(heat3, heatmap_var)
		loss4 = criterion(heat4, heatmap_var)
		loss5 = criterion(heat5, heatmap_var)
		loss6 = criterion(heat6, heatmap_var)

		loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
		train_losses.update(loss.item(), inputs.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	print('Train Loss: ', train_losses.avg)
	torch.save(model, 'cpm.pth')

	'''--------Validation--------'''
	# Validation
	print('-----------Validation-----------')
	# Validation data
	data = LSP_DATA('lsp', 'F:/Python/PyCharmWorkspace/CPM/lsp/', 8, Compose([TestResized(368)]))
	val_loader = torch.utils.data.dataloader.DataLoader(data, batch_size=8)

	model.eval()

	for j, data in enumerate(val_loader):
		inputs, heatmap, centermap = data

		inputs = inputs.cuda()
		heatmap = heatmap.cuda(async=True)
		centermap = centermap.cuda(async=True)

		input_var = torch.autograd.Variable(inputs)
		heatmap_var = torch.autograd.Variable(heatmap)
		centermap_var = torch.autograd.Variable(centermap)

		heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, centermap_var)

		loss1 = criterion(heat1, heatmap_var)
		loss2 = criterion(heat2, heatmap_var)
		loss3 = criterion(heat3, heatmap_var)
		loss4 = criterion(heat4, heatmap_var)
		loss5 = criterion(heat5, heatmap_var)
		loss6 = criterion(heat6, heatmap_var)

		loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
		val_losses.update(loss.item(), inputs.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	print('Validation Loss: ', val_losses.avg)
	if val_losses.avg < min_losses:
		# Save best model
		model.save(model, 'best_cpm.pth')

	model.train()

	epoch += 1
