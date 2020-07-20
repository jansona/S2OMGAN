import os, glob, cv2, time, torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, util as v_util
from util import html
import numpy as np
import json

from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
import sys
# import util

if __name__ == '__main__':
    
    # region get options from a json file
    # get_opt_json()
    # endregion

    sys.argv.extend(['--gpu_ids', '-1'])
    opt = TestOptions().parse()  # get test options
    opt.name = "demo_pretrained"
#     opt.epoch = 200
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    #opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.load_size = opt.crop_size
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
#     model.setup(opt)               # regular setup: load and print networks; create schedulers

    load_path = opt.MODEL_FILE
    net = getattr(model, 'netG_A')
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    print('loading the model from %s' % load_path)
    state_dict = torch.load(load_path, map_location=str(model.device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        model._BaseModel__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    net.load_state_dict(state_dict)
    model.print_networks(opt.verbose)

    starttime = time.time()
    lasttime = starttime

    if opt.eval:
        model.eval()
    starttime = time.time()
    
    print(opt.IMAGE_PATH, opt.RESULT_PATH)
    input_img = Image.open(opt.IMAGE_PATH).convert('RGB')
    transform_params = get_params(opt, input_img.size)
    transformer = get_transform(opt, transform_params, grayscale=(opt.input_nc == 1))
    x = transformer(input_img)
#     print(type(x), x.shape, x)
    
    model.real_A = x.unsqueeze(0).to(model.device)
    model.forward()
    x = model.fake_B[0]
#     print(type(x), x.shape, x)

    x = (np.transpose(x.detach().numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0
    x = x.astype(np.uint8)

    outimg = Image.fromarray(x)
    outimg = outimg.resize(input_img.size)
    outimg.save( opt.RESULT_PATH)
