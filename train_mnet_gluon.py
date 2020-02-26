# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx
import mxnet.gluon.model_zoo.vision as vision

def set_imagenet_aug(aug):
    # standard data augmentation setting for imagenet training
    aug.set_defaults(rgb_mean='123.68,116.779,103.939', rgb_std='58.393,57.12,57.375')
    aug.set_defaults(random_crop=0, random_resized_crop=1, random_mirror=1)
    aug.set_defaults(min_random_area=0.08)
    aug.set_defaults(max_random_aspect_ratio=4./3., min_random_aspect_ratio=3./4.)
    aug.set_defaults(brightness=0.4, contrast=0.4, saturation=0.4, pca_noise=0.1)

if __name__ == '__main__':
    train_fname = "mydata_train.rec"
    val_fname ="mydata_val.rec"

    # parse args
    parser = argparse.ArgumentParser(description="train kouzhao",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    parser.set_defaults(
        # network
        network          = 'mnet',
        
        #data
        use_imagenet_data_augmentation = True,
        data_train     = train_fname,
        data_val       = val_fname,
        num_examples     = 6120,
        image_shape      = '3,128,128',
        min_random_scale = 1, # if input image has min size k, suggest to use
                              # 256.0/x, e.g. 0.533 for 480
        # train
        gpus='0',
        num_epochs       = 80,
        lr_step_epochs   = '30,60',
        dtype            = 'float32'
    )
    args = parser.parse_args()
    if args.use_imagenet_data_augmentation:
        set_imagenet_aug(parser)

    # load network
    #from importlib import import_module
    #net = import_module('symbols.mobilenetv2')
    #sym = net.get_symbol(num_classes=2, multiplier=1.0)
    #sym = vision.mobilenet0_25(pretrained = False, ctx = mx.gpu(0))
    sym = vision.get_model('mobilenetv2_0.25', pretrained = False, classes = 2, ctx = mx.gpu(0))
    # train
    fit.fit(args, sym, data.get_rec_iter)
