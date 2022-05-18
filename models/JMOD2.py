import os
import sys
import time
import math
import numpy as np
import logging
import tensorflow as tf
import tensorflow.keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
#from keras.backend.tensorflow_backend import set_session
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Reshape, Convolution2D, Input, Conv2DTranspose, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
import sys
sys.path.append('.')

from lib.DataAugmentationStrategy import DataAugmentationStrategy
from lib.SampleType import Sample_Architecture_V1, Sample_Architecture_V2
from lib.DataGenerationStrategy import SampleGenerationStrategy
from lib.Dataset import Dataset
from lib.DepthCallback import PrintBatch, TensorBoardCustom
from lib.DepthMetrics import *
from lib.DepthObjectives import log_normals_loss
from lib.ObstacleDetectionObjectives import *
from lib.EvaluationUtils import get_detected_obstacles_from_detector_v1, get_detected_obstacles_from_detector_v2



class JMOD2(object):
    def __init__(self, config):
        # config data
        self.config = config
        self.dataset = {}
        self.training_set = {}
        self.test_set = {}
        self.validation_set = {}
        self.data_augmentation_strategy = DataAugmentationStrategy()
        self.shuffle = True
        # Model
        if self.config.version_model == 1:
            self.batch_norm_flag = self.config.batch_normalization
            self.model = self.build_model_v1()
        else:
            self.batch_norm_flag = self.config.batch_normalization
            self.model = self.build_model_v2()
        # Prepare dataset
        self.prepare_data()

    def load_dataset(self):
        if self.config.version_model == 1:
            dataset = Dataset(self.config, SampleGenerationStrategy(sample_type=Sample_Architecture_V1))
        else:
            dataset = Dataset(self.config, SampleGenerationStrategy(sample_type=Sample_Architecture_V2))
        dataset_name = 'UnrealDataset'
        return dataset, dataset_name

    def prepare_data(self):
        dataset, dataset_name = self.load_dataset()
        # Read dir
        self.dataset[dataset_name] = dataset
        logging.info("Loading dataset {}".format(dataset_name))
        self.dataset[dataset_name].read_data()
        # Training and test dir
        self.training_set, self.test_set = self.dataset[dataset_name].generate_train_test_data()
        # Get Validation set
        np.random.shuffle(self.training_set)
        train_val_split_idx = int(len(self.training_set)*(1-self.config.validation_split))
        self.validation_set = self.training_set[train_val_split_idx:]
        self.training_set = self.training_set[0:train_val_split_idx]

    def prepare_data_for_model(self, features, label):
        # Normalize input
        features = np.asarray(features)
        features = features.astype('float32')
        features /= 255.0
        # Prepare output : list of numpy arrays
        labels_depth = np.zeros(shape=(features.shape[0],features.shape[1],features.shape[2],1), dtype=np.float32) # Gray Scale
        if self.config.version_model == 1:
            labels_obs = np.zeros(shape=(features.shape[0],40,7), dtype=np.float32) # Obstacle output
        else:
            labels_obs = np.zeros(shape=(features.shape[0],5,8,2,7), dtype=np.float32) # Obstacle output
        i = 0
        for elem in label:
            elem["depth"] = np.asarray(elem["depth"]).astype(np.float32)
            # Change depth map in 8bits to meters
            elem["depth"] = -4.586e-09 * (elem["depth"] ** 4) + 3.382e-06 * (elem["depth"] ** 3) - 0.000105 * (elem["depth"] ** 2) + 0.04239 * elem["depth"] + 0.04072
            elem["depth"] /= 39.75 # scale 0 to 1
            labels_depth[i, ...] = elem["depth"]
            labels_obs[i, ...] = np.asarray(elem["obstacles"]).astype(np.float32)
            i +=1
        return features, [labels_depth,labels_obs]

    def train_data_generator(self):
        # Shuffle
        if self.shuffle:
            np.random.shuffle(self.training_set)
        curr_batch = 0
        self.training_set = list(self.training_set)
        while 1:
            if (curr_batch + 1) * self.config.batch_size > len(self.training_set):
                np.random.shuffle(self.training_set)
                curr_batch = 0
            x_train = []
            y_train = []
            for sample in self.training_set[curr_batch * self.config.batch_size: (curr_batch + 1) * self.config.batch_size]:
                # Get input
                features = sample.read_features()
                # Get output
                label = sample.read_labels()
                if self.data_augmentation_strategy is not None:
                    features, label = self.data_augmentation_strategy.process_sample(features, label, is_test=False)
                # Append in batch list
                x_train.append(features)
                y_train.append(label)
            x_train, y_train = self.prepare_data_for_model(x_train, y_train)
            curr_batch += 1
            yield x_train , y_train

    def validation_data_generator(self):
        if self.shuffle:
            np.random.shuffle(self.validation_set)
        curr_batch = 0
        while 1:
            if (curr_batch + 1) * self.config.batch_size > len(self.validation_set):
                np.random.shuffle(self.validation_set)
                curr_batch = 0
            x_train = []
            y_train = []
            for sample in self.validation_set[curr_batch * self.config.batch_size: (curr_batch + 1) * self.config.batch_size]:
                features = sample.read_features()
                label = sample.read_labels()
                if self.data_augmentation_strategy is not None:
                    features, label = self.data_augmentation_strategy.process_sample(features, label, is_test=True)
                x_train.append(features)
                y_train.append(label)
            x_train, y_train = self.prepare_data_for_model(x_train, y_train)
            curr_batch += 1
            yield x_train , y_train

    def tensorboard_data_generator(self, num_samples):
        # Sample list
        curr_train_sample_list      = self.training_set[0:num_samples]
        curr_validation_sample_list = self.validation_set[0: num_samples]
        # Aux
        x_train = []
        y_train = []
        for sample in curr_train_sample_list:
            features = sample.read_features()
            label = sample.read_labels()
            if self.data_augmentation_strategy is not None:
                features, label = self.data_augmentation_strategy.process_sample(features, label)
            x_train.append(features)
            y_train.append(label)
        # Sample validation
        for sample in curr_validation_sample_list:
            features = sample.read_features()
            label = sample.read_labels()
            if self.data_augmentation_strategy is not None:
                features, label = self.data_augmentation_strategy.process_sample(features, label, is_test=True)
            x_train.append(features)
            y_train.append(label)
        # Prepare
        x_train, y_train = self.prepare_data_for_model(x_train, y_train)
        return x_train , y_train

    def build_depth_model(self):
        # Define input
        input = Input(shape=(self.config.input_height, self.config.input_width, self.config.input_channel),
                        name='input')
        # Features red
        vgg19model = VGG19(include_top=False, weights=None, input_tensor=input,
                        input_shape=(self.config.input_height, self.config.input_width, self.config.input_channel))
        # No use last layer
        vgg19model.layers.pop()
        # Last layer output
        output = vgg19model.layers[-2].output
        # Depth part
        x = Conv2DTranspose(128, (4, 4), padding="same", strides=(2, 2))(output)
        x = PReLU()(x)
        x = Conv2DTranspose(64, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        x = Conv2DTranspose(32, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        x = Conv2DTranspose(16, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        out = Convolution2D(1, (5, 5), padding="same", activation="relu", name="depth_output")(x)
        # Depth model
        model = Model(inputs=input, outputs=out)
        return model

    def conv_block(self, input_tensor, n_filters, filter_shape=(3, 3), act_func='relu', batch_norm=False, block_num=None):
        output_tensor = Convolution2D(n_filters, filter_shape, activation=act_func, padding='same', name='det_conv'+block_num)(input_tensor)
        if batch_norm:
            output_tensor = BatchNormalization(name='norm'+block_num)(output_tensor)
            output_tensor = LeakyReLU(alpha=0.1)(output_tensor)
        return output_tensor

    def build_model_v1(self):
        # Depth model
        depth_model = self.build_depth_model()
        # Detection model
        output = depth_model.layers[-10].output
        # Detection layers
        x = self.conv_block(output, 512, batch_norm=self.batch_norm_flag, block_num='1')
        x = self.conv_block(x, 512, batch_norm=self.batch_norm_flag, block_num='2')
        x = self.conv_block(x, 512, batch_norm=self.batch_norm_flag, block_num='3')
        x = self.conv_block(x, 512, batch_norm=self.batch_norm_flag, block_num='4')
        x = self.conv_block(x, 512, batch_norm=self.batch_norm_flag, block_num='5')
        x = self.conv_block(x, 280, batch_norm=self.batch_norm_flag, block_num='6')
        x = Reshape((40, 7, 160))(x)
        x = self.conv_block(x, 160, batch_norm=self.batch_norm_flag, block_num='7')
        x = self.conv_block(x, 40, batch_norm=self.batch_norm_flag, block_num='8')
        x = self.conv_block(x, 1, act_func='linear', batch_norm=self.batch_norm_flag, block_num='9')
        out_detection = Reshape((40, 7), name='detection_output')(x)

        model = Model(inputs=depth_model.inputs[0], outputs=[depth_model.outputs[0], out_detection])
        opt = Adam(learning_rate=self.config.learning_rate, clipnorm = 1.)
        model.compile(loss={'depth_output': log_normals_loss, 'detection_output':yolo_v1_loss},
                        optimizer=opt,
                        metrics={'depth_output': [rmse_metric, logrmse_metric, sc_inv_logrmse_metric],
                                'detection_output': [iou_metric, recall, precision, mean_metric, variance_metric,
                                yolo_objconf_loss, yolo_nonobjconf_loss, yolo_xy_loss, yolo_wh_loss,
                                yolo_mean_loss, yolo_var_loss]},
                                loss_weights=[1.0, 1.0])
        return model

    def build_model_v2(self):
        # Depth model
        depth_model = self.build_depth_model()
        #Detection section
        output = depth_model.layers[-10].output
        # Detection layers
        x = MaxPooling2D(pool_size=(2, 2), name='det_maxpool')(output)
        x = self.conv_block(x, 512, batch_norm=self.batch_norm_flag, block_num='1')
        x = self.conv_block(x, 1024, filter_shape=(1, 1), batch_norm=self.batch_norm_flag, block_num='2')
        x = self.conv_block(x, 512, batch_norm=self.batch_norm_flag, block_num='3')
        x = self.conv_block(x, 1024, filter_shape=(1, 1), batch_norm=self.batch_norm_flag, block_num='4')
        x = self.conv_block(x, 512, batch_norm=self.batch_norm_flag, block_num='5')
        x = self.conv_block(x, 256, batch_norm=self.batch_norm_flag, block_num='6')
        x = self.conv_block(x, 128, batch_norm=self.batch_norm_flag, block_num='7')
        x = self.conv_block(x, 64, batch_norm=self.batch_norm_flag, block_num='8')

        # Output detection
        x = self.conv_block(x, 14, filter_shape=(1, 1), act_func='linear', batch_norm=self.batch_norm_flag, block_num='9')
        out_detection = Reshape((5, 8, 2, 7), name='detection_output')(x)
        # Model depth and detection
        model = Model(inputs= depth_model.inputs[0], outputs=[depth_model.outputs[0], out_detection])
        # Optimizator
        opt = Adam(learning_rate=self.config.learning_rate, clipnorm = 1.)
        model.compile(loss={'depth_output': log_normals_loss,'detection_output':yolo_v2_loss},
                        optimizer=opt,
                        metrics={'depth_output': [rmse_metric, logrmse_metric, sc_inv_logrmse_metric],
                        'detection_output': [iou_metric, recall, precision,
                                            yolo_objconf_loss, yolo_nonobjconf_loss, yolo_xy_loss, yolo_wh_loss]},
                                            loss_weights=[1.0, 0.1])
        return model

    def train(self, initial_epoch=0):
        # Save model summary in a file
        orig_stdout = sys.stdout
        f = open(os.path.join(self.config.model_dir, 'model_summary.txt'), 'w')
        sys.stdout = f
        print(self.model.summary())

        # Print layers in model summary.txt
        for layer in self.model.layers:
            print(layer.get_config())
        sys.stdout = orig_stdout
        f.close()
        # Save img model summaty
        plot_model(self.model, show_shapes=True, to_file=os.path.join(self.config.model_dir, 'model_structure.pdf'))
        # Inicial time
        t0 = time.time()
        # Samples per epoch
        logging.info("Data in our training set {}".format(len(self.training_set)))
        samples_per_epoch = int(math.floor(len(self.training_set) / self.config.batch_size))
        # Validation steps
        val_step = int(math.floor(len(self.validation_set) / self.config.batch_size))
        # TODO
        # Callbacks
        # pb = PrintBatch()
        # tb_x, tb_y = self.tensorboard_data_generator(self.config.max_image_summary)
        # tb = TensorBoardCustom(self.config, tb_x, tb_y, self.config.tensorboard_dir)
        # model_checkpoint = ModelCheckpoint(os.path.join(self.config.model_dir, 'weights-{epoch:02d}-{loss:.2f}.hdf5'),monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False,mode='min', period=self.config.log_step) # Save weights every 20 epoch
        model_checkpoint = ModelCheckpoint(os.path.join(self.config.model_dir, 'weights-{epoch:02d}-{loss:.2f}.hdf5'),monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False,mode='min', save_freq=int(self.config.log_step * samples_per_epoch)) # Save weights every 20 epoch
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=60)
        # Train
        history = self.model.fit(self.train_data_generator(),
                            steps_per_epoch=samples_per_epoch,
                            #callbacks=[pb, model_checkpoint, tb, es],
                            #validation_data=self.validation_data_generator(),
                            #validation_steps=val_step,
                            epochs=self.config.num_epochs,
                            verbose=2,
                            initial_epoch=initial_epoch)
        # Final time
        t1 = time.time()
        print("Training completed in " + str(t1 - t0) + " seconds")
        return history

    def resume_training(self, weights_file, initial_epoch):
        # Load weights
        self.model.load_weights(weights_file)
        history = self.train(initial_epoch)
        return history

    def compute_correction_factor(self, depth, obstacles):
        mean_corr = 0
        it = 0
        for obstacle in obstacles:
            top = np.max([obstacle.y, 0])
            bottom = np.min([obstacle.y + obstacle.h, depth.shape[1]])
            left = np.max([obstacle.x, 0])
            right = np.min([obstacle.x + obstacle.w, depth.shape[2]])
            depth_roi = depth[0, top:bottom,left:right, 0]
            if len(depth_roi) > 0:
                mean_corr += obstacle.depth_mean / np.mean(depth_roi)
                it += 1
        # average factor
        if it > 0:
            mean_corr /= it
        else:
            mean_corr = 1.
        return mean_corr

    def compute_correction_factor_depth_ground(self, depth, ground_depth_map, obstacles):
        mean_corr = 0
        it = 0
        for obstacle in obstacles:
            bottom = np.min([obstacle.y + obstacle.h, depth.shape[1]-1])
            center = obstacle.x + (obstacle.w/2)
            if center > depth.shape[2]/2:
                corner = np.max([obstacle.x - 2, 0])
            else:
                corner = np.min([obstacle.x + obstacle.w + 2, depth.shape[2]-1])
            if ground_depth_map[bottom, corner] < 3 and obstacle.w < 256:
                mean_corr += ground_depth_map[bottom, corner] / depth[0, bottom, corner, 0] #np.mean(depth_roi)
                it += 1
        # average factor
        if it > 0:
            mean_corr /= it
        else:
            mean_corr = 1.0
        return mean_corr

    def run(self, input, ground_plane_depth_map=None, evaluate_indoors=False):
        if evaluate_indoors:
            mean = 0#np.load('Indoors_RGB_mean.npy')
        else:
            mean = np.load('Unreal_RGB_mean.npy')
        # Correct input
        if len(input.shape) == 2 or input.shape[2] == 1:
            tmp = np.zeros(shape=(input.shape[0],input.shape[1],3))
            tmp[:,:,0] = input
            tmp[:,:,1] = input
            tmp[:,:,2] = input
            input = tmp
        if len(input.shape) == 3:
            input = np.expand_dims(input-mean/255., 0)
        else:
            input[0,:,:,:] -= mean/255.

        # Get prediction
        t0 = time.time()
        net_output = self.model.predict(input)
        print ("Elapsed time: {}").format(time.time() - t0)

        # Depth map
        if evaluate_indoors:
            pred_depth = net_output[0]
        else:
            pred_depth = net_output[0] * 39.75
        # Obstacles
        pred_detection = net_output[1]
        if self.config.version_model == 1:
            pred_obstacles, rgb_with_detection = get_detected_obstacles_from_detector_v1(pred_detection, self.config.detector_confidence_thr)
        else:
            pred_obstacles, rgb_with_detection = get_detected_obstacles_from_detector_v2(pred_detection, self.config.detector_confidence_thr)
        # Depth map corrected
        correction_factor = self.compute_correction_factor(pred_depth, pred_obstacles)
        corrected_depth = np.array(pred_depth) * correction_factor
        return [pred_depth, pred_obstacles, corrected_depth]
