import cv2
import numpy as np
import lib.EvaluationUtils as EvaluationUtils
from config import get_config
import os
from glob import glob
from lib.Evaluators import JMOD2Stats

def preprocess_data(rgb, w=256, h=160):
    rgb = np.asarray(rgb, dtype=np.float32) / 255.
    rgb = np.expand_dims(rgb, 0)
    return rgb

#edit config.py as required
config, unparsed = get_config()

#Edit model_name to choose model between ['jmod2','cadena','detector','depth','eigen']
model_name = 'jmod2'

showImages = True

model, detector_only = EvaluationUtils.load_model(model_name, config)

dataset_main_dir = config.data_set_dir
test_dirs = config.data_test_dirs

# Counters
true_positive = 0
false_positive = 0
false_negative = 0
mean_iou = 0

for test_dir in test_dirs:
    rgb_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, '*' + '.jpg')))
    index = 0

    for rgb_path in rgb_paths:
        index_str = str(index)
        rgb_raw = cv2.imread(rgb_path)
        rgb_raw = cv2.resize(rgb_raw, (config.input_width, config.input_height), cv2.INTER_LINEAR)

        #Normalize input between 0 and 1, resize if required
        rgb = preprocess_data(rgb_raw, w=config.input_width, h=config.input_height)

        #Forward pass to the net
        results = model.run(rgb, evaluate_indoors=True)

        #Eliminate multiple detections
        if config.non_max_suppresion:
            best_detections_list = EvaluationUtils.non_maximal_suppresion(results[1], iou_thresh=0.3)
            results[1] = best_detections_list
        else:
            best_detections_list = results[1]

        if showImages:
            if results[1] is not None:
                EvaluationUtils.show_detections(rgb_raw, results[1], None, save_dir='test/'+test_dir, file_name=index_str.zfill(5)+'.png', sleep_for=100)
            if results[0] is not None:
                EvaluationUtils.show_depth(rgb_raw, results[0], None, save_dir='test/'+test_dir, file_name=index_str.zfill(5)+'.png', max_depth=5.5, sleep_for=100)
        index += 1