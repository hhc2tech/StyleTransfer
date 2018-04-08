from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.models import load_model
from keras.models import Model
from keras.applications.vgg19 import preprocess_input
import imageio
from keras import backend as K
import argparse
import time
from scipy import optimize

# parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
# parser.add_argument('base_image_path', metavar='base', type=str,
#                     help='Path to the image to transform.')
# parser.add_argument('style_reference_image_path', metavar='ref', type=str,
#                     help='Path to the style reference image.')
# parser.add_argument('result_prefix', metavar='res_prefix', type=str,
#                     help='Prefix for the saved results.')
# parser.add_argument('--iter', type=int, default=10, required=False,
#                     help='Number of iterations to run.')
# parser.add_argument('--content_weight', type=float, default=0.025, required=False,
#                     help='Content weight.')
# parser.add_argument('--style_weight', type=float, default=1.0, required=False,
#                     help='Style weight.')
# parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
#                     help='Total Variation weight.')
#
# args = parser.parse_args()
# base_image_path = args.base_image_path
# style_reference_image_path = args.style_reference_image_path
# result_prefix = args.result_prefix
# iterations = args.iter

content_image_path = r'G:/Keras/neural_style/Data/content/lion.jpg'
style_image_path = r'G:/Keras\neural_style\Data\style\starry-night.jpg'
generated_image_shape = (1, 224, 224, 3)
img_rows = img_columns = 224
img_channels = 3
result_prefix = 'tuwei_cnn'
iterations = 10
alpha = 0.1
beta = 100
# def model():
#     neural_model = VGG19(weights='imagenet')
#     #neural_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
#     return neural_model


def preprocess_img(img_path=r'G:/Keras/neural_style/Data/content/lion.jpg'):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_rows, img_columns))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_rows, img_columns, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def compute_content_cost(img_content, img_generate):
    height, width, channel = K.int_shape(img_content)
    J_content = K.sum(K.square(img_content - img_generate)) / (4 * height * width * channel)
    return J_content


def compute_gram(a):
    assert K.ndim(a) ==3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(a)
    else:
        features = K.batch_flatten(K.permute_dimensions(a, (2, 0, 1)))
    return K.dot(features, K.transpose(features))


def compute_layer_style_lost(img_style, img_generate):
    height, width, channels = K.int_shape(img_style)
    S = compute_gram(img_style)
    G = compute_gram(img_generate)
    return K.sum(K.square(S-G))/(4*height**2*width**2*channels**2)



content_image = K.variable(preprocess_img(content_image_path))
style_image = K.variable(preprocess_img(style_image_path))
generated_image = K.placeholder(shape=generated_image_shape)
input_tensor = K.concatenate([content_image, style_image, generated_image], axis=0)
model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
feature_layers = {'block1_conv1': 0.2, 'block2_conv1': 0.2,
                  'block3_conv1': 0.2, 'block4_conv1': 0.2,
                  'block5_conv1': 0.2}
output_dict = {layer.name: layer.output for layer in model.layers}
layer_fetures = output_dict['block5_conv2']
content_image_feature = layer_fetures[0, :, :, :]
style_image_feature = layer_fetures[1, :, :, :]
generated_image_feature = layer_fetures[2, :, :, :]
loss = K.variable(0.0)
J_content = compute_content_cost(content_image_feature, generated_image_feature)
loss += alpha*J_content
J_style = 0
for layer_name in feature_layers.keys():
    layer_fetures = output_dict[layer_name]
    style_features = layer_fetures[1, :, :, :]
    generated_features = layer_fetures[2, :, :, :]
    J_layer = compute_layer_style_lost(style_features, generated_features)
    J_style += J_layer * feature_layers[layer_name]
loss += beta*J_style

grads = K.gradients(loss,  generated_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)
f_outputs = K.function([generated_image], outputs)

def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_rows, img_columns))
    else:
        x = x.reshape((1, img_rows, img_columns, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

x = preprocess_img(content_image_path)

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = optimize.fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())
    fname = r'G:/Keras/neural_style/Data/output/' + result_prefix + '_at_iteration_%d.png' % i
    imageio.imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
