"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    Test a SOMGAN model:
        python test.py --dataroot datasets/maps --name maps_somgan --model somgan

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os, glob, cv2, time, torch, math
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
from scipy.signal import convolve2d
import json


def MSE(pic1, pic2):
    return np.sum(np.square(pic1 - pic2)) / (pic1.shape[0] * pic1.shape[1])

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

d_name = os.path.dirname(__file__)


x_size = 17
y_size = 20
zoom = 17


def num2deg(y,x,zoom):
    n=2**zoom
    lon_deg=x/n*360.0-180.0
    lat_deg=math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg=lat_deg*180.0/math.pi
    return [lon_deg,lat_deg]

def integrate_tiles(tile_mat: [[str]]) -> np.array:
    
    def assemble_row(row_files: [str]) -> np.array:
        
        tile_cated = cv2.imread(os.path.join(d_name,row_files[0]))
        
        for file in row_files[1:]:
            temp_tile = cv2.imread(os.path.join(d_name,file))
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
    y_list = []
    x_list = []
    for name in name_list:
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

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.dataroot = opt.DATA_PATH
#     opt.epoch = 200
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    #opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.load_size = opt.crop_size
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    # load model from a path - for platform
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
    
    # create a website
    web_dir = opt.OUTPUT_PATH # specific output dir - for platform
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    starttime = time.time()
    lasttime = starttime
    
    if opt.eval:
        model.eval()
    starttime = time.time()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i > 0 and (i + 1) % 10 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (len(img_path)+(i)*opt.batch_size, ''), 'cost', time.time()-lasttime, 'seconds')
            lasttime = time.time()
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
    print("Work Done!!!")
    print('Generated', len(dataset), 'maps. Total Time Cost: ', lasttime - starttime, 'seconds')

        
    # automatic integrate and autocontrast - for platform
    
    print("start integrating...")
    
    in_path = webpage.get_image_dir()
    out_path = in_path[:-6] + "integrated"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    
    x_min,x_max,y_min,y_max = statis_value(in_path)
    
    for x in range(x_min,x_max+2-x_size,x_size):
        for y in range(y_min,y_max+2-y_size,y_size):
            print("integrating tile at", x, y)
            
            tfw = ['0.0000107288','0.0000000000','0.0000000000','-0.00000811273']
            base_path = in_path + "/"
            file_template = "{}_{}_fake_B.png"
            tile_files = []
            for i in range(x_size):
                temp_list = []
                for j in range(y_size):
                    temp_list.append(base_path + file_template.format(y + j, x + i))
                    
                tile_files.append(temp_list)
        
            map_pic = integrate_tiles(tile_files)
            cv2.imwrite(os.path.join(out_path, "{}_{}.tif".format(y, x)), map_pic)
            
            nimg = AutoContrast(map_pic, 3)
            cv2.imwrite(os.path.join(out_path, "{}_{}_autocontrast.jpg".format(y, x)), nimg)
            
            [lon,lat] = num2deg(x,y,zoom)
            tfw.extend([lon,lat])
            file_tfw=open(os.path.join(out_path,"{}_{}.tfw".format(y, x)),mode='w')
            for i in range(6):
                tfw_name = str(tfw[i])+'\n'
                file_tfw.write(tfw_name)
    print ("integrated done")
                
# region 计算指标measure并生成排序文件
    def generate_ranking_file():
        result_path = os.path.join(opt.OUTPUT_PATH, "images")
        realB_names = glob.glob("{}/*_real_B.png".format(result_path))

        results_with_metric = []
        sum_metric = 0
        suffix_len = len("_real_B.png")
        for n in realB_names:
            measure = cal_measure(n)
            sum_metric += measure
            results_with_metric.append((n.split("/")[-1][:len(n)-suffix_len], measure))

        results_with_metric.sort(key=lambda x:x[1], reverse=True)

        with open("{}/images/metric_ranking.txt".format(opt.OUTPUT_PATH), "w", encoding="utf-8") as fout:
            for sample_num, sample_metric in results_with_metric:
                fout.write("{}:{}\n".format(sample_num, sample_metric))
                
        return sum_metric / len(realB_names)

    avg_metric = generate_ranking_file()
# endregion
    
    # export metrics results - for platform
    r_report = {"tables" : [{"tableName": "测试汇报", "结果": {
        "瓦片个数": len(dataset), 
        "总时长": lasttime - starttime,
        "指标": avg_metric
    }}]}
#     print(r_report)

    with open(opt.RESULT_PATH, "w") as jsonof:
        json.dump(r_report, jsonof, ensure_ascii=False)
