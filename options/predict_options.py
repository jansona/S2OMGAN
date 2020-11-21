from .base_options import BaseOptions


class PredictOptions(BaseOptions):
    """This class includes predict options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=5000, help='how many test images to run')

        # for AI platform
        parser.add_argument('--IMAGE_PATH', required=True, type=str, help='the path of input image(s)')
        parser.add_argument('--MODEL_FILE', required=True, default='map_generation.pth', type=str, help='导入模型文件路径')
        parser.add_argument('--RESULT_PATH', required=True, default='final.png', type=str, help='最终结果文件路径')
        parser.add_argument('--zoom', type=int, default=17, help='the level of geo data')
        # used in batch_generate
        parser.add_argument('--DATA_PATH', default='', type=str, help='数据源路径')
        parser.add_argument('--OUTPUT_PATH', default='generated', type=str, help='生成结果路径')

        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
