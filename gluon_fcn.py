import os
import mxnet as mx
import cv2
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
import matplotlib.pyplot as plt
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg

# using cpu
print('all imports done')

ctx = mx.cpu(0)
IM_PATH = '../tf_detection_zoo/ssd_mobilenet_v2_coco_2018_03_29/selected_images_classes'
SAVE_PATH = 'masks'

transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])

model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
print('Model loaded')


for dirs in os.listdir(IM_PATH):
	if not dirs.startswith('.'):
		for im in os.listdir(os.path.join(IM_PATH, dirs)):
			if im.endswith('.jpeg'):
				img = image.imread(os.path.join(IM_PATH, dirs, im))
				if img is None:
					print('image read as None')
				print('image name: ', im)

				img = transform_fn(img)
				img = img.expand_dims(0).as_in_context(ctx)
				print('Normalisation done')


				output = model.demo(img)
				predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
				print('prediction made')


				mask = get_color_pallete(predict, 'pascal_voc')

				saveDir = os.path.join(SAVE_PATH, dirs)

				if not os.path.exists(saveDir):
					os.mkdir(saveDir)
				finalPath = os.path.join(saveDir, im.rstrip('.jpeg') + '.png')
				#print(finalPath)
				mask.save(finalPath)
				print('mask generated and saved')

				'''
				mmask = mpimg.imread('predict.png')
				print('mask read again')

				plt.subplot(1,2,1)
				plt.imshow(mmask)

				plt.subplot(1, 2, 2)
				plt.imshow(img)
				plt.show()
				'''