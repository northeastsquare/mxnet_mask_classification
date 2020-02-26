import mxnet
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from IPython import display
import matplotlib.pyplot as plt
from mxnet.gluon.model_zoo import vision as models
from mxnet import image
import os

ctx = mxnet.gpu(0)
net = gluon.nn.SymbolBlock.imports("mobilenetv2-0.5-new-symbol.json", ['data'], \
     "mobilenetv2-0.5-new-0000.params", ctx=ctx)
net.collect_params().reset_ctx(ctx=ctx)

def transform(data):
    data = data.transpose((2,0,1)).expand_dims(axis=0)
    return data.astype('float32')
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std

#prekwargs =  {'ctx': ctx, 'pretrained': False, 'classes': 2}
# net = models.get_model("mobilenetv2_0.5", **prekwargs)
# net.load_parameters('./checkpoints/mobilenetv2_0.5_best.params', allow_missing=False, ignore_extra=False, ctx=ctx)
for root, d, files in os.walk('/home/silva/work/mask/crops/face_mask/'):
    for f in files:
        fn, ext = os.path.splitext(f)
        if ext != '.jpg':
            continue
        fname = os.path.join(root, f)
        print("fname:", f)
        #fname = "/home/silva/work/mask/crops/face/1_Handshaking_Handshaking_1_130_1.jpg"
        #fname = "/home/silva/work/mask/crops/face_mask/test_00000015_0.jpg"
        x = image.imread(fname)
        #x = image.resize_short(x, 256)
        #x, _ = image.center_crop(x, (224,224))
        # plt.imshow(x.asnumpy())
        # plt.show()
        x = x.copyto(ctx)
        #prob = net(transform(x))#.softmax()
        prob = net(x.astype('float32'))
        print("prob:", prob)   
