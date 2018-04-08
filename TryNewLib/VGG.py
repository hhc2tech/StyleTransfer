from keras.applications.vgg19 import VGG19

model = VGG19(weights='imagenet')
output_dict = {layer.name: layer.output for layer in model.layers}
output_dict['block5_conv2']