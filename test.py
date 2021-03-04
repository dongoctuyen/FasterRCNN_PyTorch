import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils import array_tool as at
import os
from PIL import ImageDraw, Image, ImageFont

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

img = read_image('misc/bs (20).jpg')
img = t.from_numpy(img)[None]
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn)

trainer.load('./checkpoints/fasterrcnn_02041242_0.7443192583154143', map_location='cpu')
opt.caffe_pretrain = False  # this model was trained from torchvision-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img)
print(_bboxes)
VQC = ('vqc',)
print(_bboxes, _labels, _scores)
img = Image.open('misc/bs (20).jpg')
new_img = ImageDraw.Draw(img)
for i in range(len(_labels[0])):
    new_img.rectangle((_bboxes[0][i][1], _bboxes[0][i][0], _bboxes[0][i][3], _bboxes[0][i][2]), outline='#ffffff',
                      width=3)
    new_img.text((_bboxes[0][i][1], _bboxes[0][i][0]), VQC[_labels[0][i]] + ': ' + str(_scores[0][i]),
                 font=ImageFont.truetype("arial.ttf", 30))
img.show()
