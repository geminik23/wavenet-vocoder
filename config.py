import os
import dotenv

dotenv.load_dotenv()



DATASET_PATH = os.environ.get('DATASET_PATH')
if DATASET_PATH is None: 
    DATASET_PATH = './data'
    os.makedirs(DATASET_PATH, exist_ok=True)

class Config:
    def __init__(self):
        # dataset
        self.dataset_path = DATASET_PATH

        self.sc_train_audio_length = 4800 # one second

        self.lj_folder_name ="LJSpeech-1.1"
        self.lj_train_audio_length = 4096 

        self.num_workers=8

        # check point 
        self.check_point_folder = "model_cp"
        self.check_point_file = "model_e_{}.pt"
        self.check_point_period = 5

        # model parameters
        self.num_class = 256 
        self.num_cycle_block= 4
        self.max_dilation = 8
        self.residual_channels = 120
        self.skip_channels = 240
        self.upsample_kernel = 800
        self.causal_kernel = 2

        # mulaw
        self.mu = self.num_class - 1


        # params for melspecs
        self.n_mels = 40
        # self.hop_size = 256
        # self.win_size = 1024
        # self.fft_size = 1024

        self.hop_size = 64
        self.win_size = 256
        self.fft_size = 256
        self.f_min = 0
        self.f_max = 2400
        self.power = 1.0
        self.sample_rate = 4800

        # training
        self.epochs = 25
        self.lr = 5e-5 
        self.batch_size = 8
        
        self.train_after = None # checkpoint_filepath
        self.train_after = "model_cp/model_e_354.pt" # checkpoint_filepath

        ##
        # generate
        self.gen_test_idx = 54
        self.gen_result_folder='results'
        self.gen_model_cp = "model_cp/model_e_354.pt"