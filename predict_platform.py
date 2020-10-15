def predict_function(params):
# if __name__ == '__main__':

    # region get options from a json file
    # get_opt_json()
    # endregion

    import os, glob, cv2, time, torch, math, re
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
    from glob import glob
    # import util


    def integrate_tiles(d_name, tile_mat: [[str]]) -> np.array:

        for line in tile_mat:
            for tile in line:
                if not os.path.exists("{}/{}".format(d_name, tile)):
                    print(d_name, tile)
        
        def assemble_row(row_files: [str]) -> np.array:
            
            tile_cated = cv2.imread(os.path.join(d_name, row_files[0]))
            
            for file in row_files[1:]:
                temp_tile = cv2.imread(os.path.join(d_name, file))
                array_temp = np.array(temp_tile)
                if array_temp.ndim == 0:
                    break
                tile_cated = np.concatenate((tile_cated, temp_tile), axis=1)
                
            return tile_cated
        
        rows = []
        
        for row in tile_mat:
            rows.append(assemble_row(row))
            
        map_cated = rows[0]
        
        for row in rows[1:]:
            map_cated = np.concatenate((map_cated, row), axis=0)
            
        return map_cated

    def statis_value(in_path):
        name_list = os.listdir(in_path)
        name_list = list(filter(lambda x: re.match("\d+_\d+.png", x), name_list))
        y_list = []
        x_list = []
        for name in name_list:
            name = name.split("/")[-1]
            name = name.split(".")[0]
            a = name.split('_',2)
            y_list.append(int(a[0]))
            x_list.append(int(a[1]))
        x_min,x_max = min(x_list),max(x_list)
        y_min,y_max = min(y_list),max(y_list)
        return x_min,x_max,y_min,y_max

    def AutoContrast(img, flag = 1):
        """RGB影像自动对比度调整
        @img 输入影像
        @flag 处理范围计算方式 1:转为灰度计算最值 2:三个波段最值的最值 3:三个波段最值均值"""
        #根据影像灰度计算调整范围
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #根据拉伸百分比决定拉伸范围
        if flag == 1:
            minp = np.min(np.min(img_gray))
            maxp = np.max(np.max(img_gray))
        elif flag == 2:
            minp = np.min(np.min(img))
            maxp = np.max(np.max(img))
        elif flag == 3:
            minn = np.zeros((1, 3), dtype = float)
            maxn = np.zeros((1, 3), dtype = float)
            for i in range(3):
                minn[0, i] = np.min(np.min(img[:, :, i]))
                maxn[0, i] = np.max(np.max(img[:, :, i]))
            minp = np.mean(minn)
            maxp = np.mean(maxn)
        else:
            raise Exception('the flag can only be 1 or 2')
        #RGB进行相同的调整
        img = img.astype(np.float)
        deart = float(maxp - minp)
        imgout = img.copy()
        imgout[:, :, 0] = 255*(img[:, :, 0] - minp) / deart
        imgout[:, :, 1] = 255*(img[:, :, 1] - minp) / deart
        imgout[:, :, 2] = 255*(img[:, :, 2] - minp) / deart
        imgout = imgout.astype(np.uint8)
        return imgout

    def num2deg(y, x, zoom):
        n = 2**zoom
        lon_deg = x/n*360.0-180.0
        lat_deg = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = lat_deg*180.0/math.pi
        return [lon_deg, lat_deg]

    sys.argv = params
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
    source_path = opt.IMAGE_PATH

    if os.path.isdir(source_path):
        opt.RESULT_PATH = opt.RESULT_PATH.split('.')[0] + '.tif'
        result_dir_path = '/'.join(opt.RESULT_PATH.split('/')[:-1])
        temp_img_paths = []
        temp_suffix_name = glob("{}/*".format(source_path))[0].split('.')[-1]
        for name_path in glob("{}/*".format(source_path)):
            img_name = name_path.split("/")[-1]
            input_img = Image.open(name_path).convert('RGB')
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
            temp_img_path = "{}/{}".format(result_dir_path, img_name)
            temp_img_paths.append(temp_img_path)
            outimg.save(temp_img_path)

        # automatic integrate and autocontrast - for platform
        print("start integrating...")
        
        # in_path = webpage.get_image_dir()
        in_path = result_dir_path 
        # out_path = in_path[:-6] + "integrated"
        out_path = in_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        x_min, x_max, y_min, y_max = statis_value(in_path)
        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1
        zoom = 17
                
        tfw = ['0.0000107288', '0.0000000000', '0.0000000000', '-0.00000811273']
        base_path = in_path + "/"
        file_template = "{:06d}_{:06d}." + temp_suffix_name
        tile_files = []
        for i in range(x_size):
            temp_list = []
            for j in range(y_size):
                temp_list.append(file_template.format(y_min + j, x_min + i))
                
            tile_files.append(temp_list)
    
        map_pic = integrate_tiles(result_dir_path, tile_files)
        cv2.imwrite(opt.RESULT_PATH, map_pic)
        
        # nimg = AutoContrast(map_pic, 3)
        # cv2.imwrite(os.path.join(out_path, "{}_{}_autocontrast.jpg".format(y_min, x_min)), nimg)
        
        [lon,lat] = num2deg(x_min, y_min, zoom)
        tfw.extend([lon,lat])
        file_tfw=open(opt.RESULT_PATH.split('.')[0] + '.tfw', mode='w')
        for i in range(6):
            tfw_name = str(tfw[i])+'\n'
            file_tfw.write(tfw_name) 

        # delete temp img
        for p in temp_img_paths:
            if os.path.exists(p):
                os.remove(p)
            else:
                print(p, 'not exists')

        print ("integrated done")

    else:
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
        outimg.save(opt.RESULT_PATH)


if __name__ == '__main__':
    import sys, json

    # json_file_name = sys.argv[1]

    # params = []
    # action = ""

    # with open(json_file_name, 'rb') as fin:
    #     opt_json = json.load(fin)

    #     if "task" in opt_json.keys():
    #         for k, v in opt_json.items():
    #             if k == 'task':
    #                 action = v
    #             else:
    #                 params.append("--{}".format(k))
    #                 params.append("{}".format(v))
    #     else:
    #         print("Please assign param 'task'")

    # predict_function(['./predict_platform.py'] + params)
    predict_function(sys.argv)
