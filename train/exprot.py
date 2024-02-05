import torch
from nets.yolo import YoloBody

model = YoloBody
checkpoint = torch.load('model_data/best_epoch_weights.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  