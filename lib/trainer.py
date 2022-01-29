import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('Agg')

class Trainer(object):
	def __init__(self, config, model, rng):
		self.config = config
		self.model = model

	def train(self):
		print("[*] Training starts...")
		history = self.model.train()
		self.plot_metrics(history)
	
	def resume_training(self):
		print("resuming training from weights file: ", self.config.weights_path)
		history = self.model.resume_training(self.config.weights_path, self.config.initial_epoch)
		self.plot_metrics(history)

	def plot_metrics(self, history):
		plt.ioff()
		# Plot Detection
		fig = plt.figure()
		plt.plot(history.history['detection_output_iou_metric'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_iou_metric'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection IoU')
		plt.xlabel('Epochs')
		plt.ylabel('IoU')
		plt.legend()
		plt.savefig('IoU.pdf')
		plt.close(fig)
		#plt.show()
		fig = plt.figure()
		plt.plot(history.history['detection_output_recall'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_recall'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection Recall')
		plt.xlabel('Epochs')
		plt.ylabel('Recall')
		plt.legend()
		plt.savefig('recall.pdf')
		plt.close(fig)
		#plt.show()
		fig = plt.figure()
		plt.plot(history.history['detection_output_precision'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_precision'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection Precision')
		plt.xlabel('Epochs')
		plt.ylabel('Precision')
		plt.legend()
		plt.savefig('precision.pdf')
		plt.close(fig)
		#plt.show()
		fig = plt.figure()
		plt.plot(history.history['detection_output_mean_metric'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_mean_metric'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection Mean')
		plt.xlabel('Epochs')
		plt.ylabel('Mean')
		plt.legend()
		plt.savefig('mean.pdf')
		plt.close(fig)
		#plt.show()
		fig = plt.figure()
		plt.plot(history.history['detection_output_variance_metric'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_variance_metric'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection Variance')
		plt.xlabel('Epochs')
		plt.ylabel('Variance')
		plt.legend()
		plt.savefig('variance.pdf')
		plt.close(fig)
		#plt.show()
		# Plot depth
		fig = plt.figure()
		plt.plot(history.history['depth_output_rmse_metric'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_depth_output_rmse_metric'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Depth RMSE (linear)')
		plt.xlabel('Epochs')
		plt.ylabel('RMSE')
		plt.legend()
		plt.savefig('rmse.pdf')
		plt.close(fig)
		#plt.show()
		fig = plt.figure()
		plt.plot(history.history['depth_output_logrmse_metric'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_depth_output_logrmse_metric'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Depth RMSE (log)')
		plt.xlabel('Epochs')
		plt.ylabel('RMSE (log)')
		plt.legend()
		plt.savefig('logrmse.pdf')
		plt.close(fig)
		#plt.show()
		fig = plt.figure()
		plt.plot(history.history['depth_output_sc_inv_logrmse_metric'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_depth_output_sc_inv_logrmse_metric'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Depth RMSE (log, scale-invariant)')
		plt.xlabel('Epochs')
		plt.ylabel('RMSE (log, scale-invariant)')
		plt.legend()
		plt.savefig('invlogrmse.pdf')
		plt.close(fig)
		#plt.show()
		# Plot loss
		fig = plt.figure()
		plt.plot(history.history['loss'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_loss'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('loss.pdf')
		plt.close(fig)
		#plt.show()
		fig = plt.figure()
		plt.plot(history.history['depth_output_loss'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_depth_output_loss'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Depth Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('lossdepth.pdf')
		plt.close(fig)
		#plt.show()
		fig = plt.figure()
		plt.plot(history.history['detection_output_loss'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_loss'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('lossdetection.pdf')
		plt.close(fig)
		#plt.show()
		# yolo loss 
		fig = plt.figure()
		plt.plot(history.history['detection_output_yolo_objconf_loss'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_yolo_objconf_loss'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection Loss Confidence (obstacle)')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('yolo_loss_conf.pdf')
		plt.close(fig)
		# yolo loss no obj
		fig = plt.figure()
		plt.plot(history.history['detection_output_yolo_nonobjconf_loss'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_yolo_nonobjconf_loss'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection Loss Confidence (no obstacle)')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('yolo_loss_conf_noobj.pdf')
		plt.close(fig)
		# yolo loss xy
		fig = plt.figure()
		plt.plot(history.history['detection_output_yolo_xy_loss'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_yolo_xy_loss'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection Loss Location')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('yolo_xy_loss.pdf')
		plt.close(fig)
		# yolo loss wh
		fig = plt.figure()
		plt.plot(history.history['detection_output_yolo_wh_loss'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_yolo_wh_loss'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection Loss Size')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('yolo_wh_loss.pdf')
		plt.close(fig)
		# yolo loss mean depth
		fig = plt.figure()
		plt.plot(history.history['detection_output_yolo_mean_loss'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_yolo_mean_loss'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection Loss Mean Depth')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('yolo_mean_loss.pdf')
		plt.close(fig)
		# yolo loss var depth
		fig = plt.figure()
		plt.plot(history.history['detection_output_yolo_var_loss'], label = 'Training value', color = 'darkslategray')
		plt.plot(history.history['val_detection_output_yolo_var_loss'], label = 'Validation value', color = 'darkslategray', linestyle = '--')
		plt.title('Detection Loss Variance Depth')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('yolo_var_loss.pdf')
		plt.close(fig)

		# convert the history.history dict to a pandas DataFrame:     
		hist_df = pd.DataFrame(history.history)

		# or save to csv: 
		hist_csv_file = 'history.csv'
		with open(hist_csv_file, mode='w') as f:
			hist_df.to_csv(f)