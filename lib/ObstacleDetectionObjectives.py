import tensorflow as tf
import keras.backend as K
import numpy as np
from .DepthMetrics import rmse_metric

def numpy_overlap(x1, w1, x2, w2):
    l1 = (x1) - w1 / 2
    l2 = (x2) - w2 / 2
    left = np.where(l1>l2, l1, l2)
    r1 = (x1) + w1 / 2
    r2 = (x2) + w2 / 2
    right = np.where(r1>r2, r2, r1)
    result = (right - left)
    return result

def numpy_iou(centroid_1, centroid_2, dims_1, dims_2):
    ow = numpy_overlap(centroid_1[0], dims_1[0] , centroid_2[0] , dims_2[0])
    oh = numpy_overlap(centroid_1[1], dims_1[1] , centroid_2[1] , dims_2[1])
    # compute intersection
    ow = np.where(ow > 0, ow, 0)
    oh = np.where(oh > 0, oh, 0)
    intersection = ow * oh
    # compute union
    area_1 = dims_1[0] * dims_1[1]
    area_2 = dims_2[0] * dims_2[1]
    union = area_1 + area_2 - intersection
    # iou=intersection/union
    pred_iou = intersection / (union + 1e-6) # prevent div 0
    # is it total overlap?
    minor_area = np.where(area_2 < area_1, area_2, area_1)
    is_overlap = np.where(minor_area == intersection, 1, 0)
    if pred_iou < 0:
        return 0
    return pred_iou, is_overlap

def iou(y_true, y_pred):
	size = (tf.shape(y_true)).shape
	# Adjust prediction
	# adjust x, y
	pred_xy_tensor = K.sigmoid(y_pred[..., 1:3]) # relative to position to the containing cell

	# adjust w, h
	if size == 5:
		# Anchors
		anchors = np.array([[0.34755122, 0.84069513],
							[0.14585618, 0.25650666]], dtype=np.float32)
		pred_wh_tensor = K.exp(y_pred[..., 3:5]) * K.reshape(anchors, [1, 1, 1, 2, 2]) # relative to image shape
	else:
		pred_wh_tensor = K.sigmoid(y_pred[..., 3:5]) # relative to image shape

	# adjust confidence
	pred_conf_tensor = K.sigmoid(y_pred[..., 0])

	# Adjust true tensor
	# adjust x, y
	true_xy_tensor = y_true[..., 1:3] # relative position to the containing cell

	# adjust w, h
	true_wh_tensor = y_true[..., 3:5] # relative to image shape

	# grid dimentions: x, y
	grid = [256., 160.]
	rows = []
	cols = []
	for y in range(5):
		for x in range(8):
			cell = [x, y]
			if size == 5:
				offset = [cell, cell]
			else:
				offset = [cell]
			cols.append(offset)
		rows.append(cols)
		cols = []
	if size == 5:
		offset = K.reshape(tf.convert_to_tensor(rows, dtype=np.float32), [1, 5, 8, 2, 2])
	else:
		offset = K.reshape(tf.convert_to_tensor(rows, dtype=np.float32), [1, 40, 2])

	# adjust true min and max
	if size == 5:
		true_wh_cell = true_wh_tensor * K.reshape([grid, grid], [1, 1, 1, 2, 2])
	else:
		true_wh_cell = true_wh_tensor * K.reshape([grid], [1, 1, 2])
	true_wh_half = 0.5 * true_wh_cell
	true_xy_center = tf.math.add(true_xy_tensor, offset)
	true_mins = true_xy_center - true_wh_half
	true_maxes = true_xy_center + true_wh_half

	# adjust pred min and max
	if size == 5:
		pred_wh_cell = pred_wh_tensor * K.reshape([grid, grid], [1, 1, 1, 2, 2])
	else:
		pred_wh_cell = pred_wh_tensor * K.reshape([grid], [1, 1, 2])
	pred_wh_half = 0.5 * pred_wh_cell
	pred_xy_center = tf.math.add(pred_xy_tensor, offset)
	pred_mins = pred_xy_center - pred_wh_half
	pred_maxes = pred_xy_center + pred_wh_half

	# adjust intersection
	intersect_min = tf.maximum(pred_mins, true_mins)
	intersect_max = tf.minimum(pred_maxes, true_maxes)
	intersect_wh = tf.maximum(intersect_max - intersect_min, 0.0)
	intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

	# adjust areas
	true_areas = true_wh_cell[..., 0] * true_wh_cell[..., 1]
	pred_areas = pred_wh_cell[..., 0] * pred_wh_cell[..., 1]
	union_areas = true_areas + pred_areas - intersect_areas
	iou_scores  = tf.truediv(intersect_areas, union_areas) # shape(None, 5, 8, 5)
	return iou_scores

def iou_metric(y_true, y_pred):
	# call iou
	iou_scores = iou(y_true, y_pred)
	# mask
	conf_mask = K.expand_dims(y_true[..., 0], axis=-1)
	# iou mean
	ave_iou = tf.reduce_sum(K.expand_dims(iou_scores, axis=-1) * conf_mask) / (tf.reduce_sum(conf_mask) + K.epsilon())
	return ave_iou

def recall(y_true, y_pred):
	# call iou
	iou_scores = iou(y_true, y_pred)
	# mask
	conf_mask = K.expand_dims(y_true[..., 0], axis=-1)
	# counter: true positives + false negatives
	true_p_false_n = tf.reduce_sum(conf_mask)
	# counter: true positives + false positives
	band_conf = K.greater(K.sigmoid(y_pred[..., 0]), 0.5)
	# counter: true positives
	band_iou = K.greater(iou_scores, 0.5)
	band_tp = tf.logical_and(band_conf, band_iou)
	true_p = K.sum(tf.where(band_tp, K.ones_like(y_pred[..., 0]), K.zeros_like(y_pred[..., 0])))
	# Recall: true positive / (true positives + false negatives)
	recall = true_p / (true_p_false_n + 0.00000001)
	return recall

def precision(y_true, y_pred):
	# call iou
	iou_scores = iou(y_true, y_pred)
	# counter: true positives + false positives
	band_conf = K.greater(K.sigmoid(y_pred[..., 0]), 0.5)
	true_p_false_p = tf.reduce_sum(tf.where(band_conf, K.ones_like(y_pred[..., 0]), K.zeros_like(y_pred[..., 0])))
	# counter: true positives
	band_iou = K.greater(iou_scores, 0.5)
	band_tp = tf.logical_and(band_conf, band_iou)
	true_p = K.sum(tf.where(band_tp, K.ones_like(y_pred[..., 0]), K.zeros_like(y_pred[..., 0])))
	# Precision: true_positives / (true_positive + false_positives)
	precision = true_p / (true_p_false_p + 0.0000001)
	return precision

def mean_metric(y_true, y_pred):
	conf_mask = K.expand_dims(y_true[..., 0], axis=-1)
	truth_m_tensor = K.expand_dims(y_true[..., 5], axis=-1)
	pred_m_tensor = K.expand_dims(y_pred[..., 5], axis=-1) * conf_mask
	return rmse_metric(truth_m_tensor, pred_m_tensor)

def variance_metric(y_true, y_pred):
	conf_mask = K.expand_dims(y_true[..., 0], axis=-1)
	truth_v_tensor = K.expand_dims(y_true[..., 6], axis=-1)
	pred_v_tensor = K.expand_dims(y_pred[..., 6], axis=-1) * conf_mask
	return rmse_metric(truth_v_tensor, pred_v_tensor)

def yolo_v1_loss(y_true, y_pred):
	# Adjust prediction
	# adjust x, y
	pred_xy_tensor = K.sigmoid(y_pred[:, :, 1:3]) # relative to position to the containing cell

	# adjust w, h
	pred_wh_tensor = K.sigmoid(y_pred[:, :, 3:5]) # relative to image shape

	# adjust confidence
	pred_conf_tensor = K.sigmoid(y_pred[:, :, 0])

	# adjust mean
	pred_m_tensor = K.expand_dims(y_pred[:, :, 5], axis=-1)

	# adjust variance
	pred_v_tensor = K.expand_dims(y_pred[:, :, 6], axis=-1)

	# Adjust true tensor
	# adjust x, y
	true_xy_tensor = y_true[:, :, 1:3] # relative position to the containing cell

	# adjust w, h
	true_wh_tensor = y_true[:, :, 3:5] # relative to image shape

	# adjust mean
	true_m_tensor = K.expand_dims(y_true[:, :, 5], axis=-1)

	# adjust variance
	true_v_tensor = K.expand_dims(y_true[:, :, 6], axis=-1)

	# Compute confidence
	# grid dimentions: x, y
	grid = [256., 160.]
	rows = []
	cols = []
	for y in range(5):
		for x in range(8):
			cell = [x, y]
			offset = [cell]
			cols.append(offset)
		rows.append(cols)
		cols = []
	offset = K.reshape(tf.convert_to_tensor(rows, dtype=np.float32), [1, 40, 2])

    # adjust true min and max
	true_wh_cell = true_wh_tensor * K.reshape([grid], [1, 1, 2])
	true_wh_half = 0.5 * true_wh_cell
	true_xy_center = tf.math.add(true_xy_tensor, offset) * 32.
	true_mins = true_xy_center - true_wh_half
	true_maxes = true_xy_center + true_wh_half

	# adjust pred min and max
	pred_wh_cell = pred_wh_tensor * K.reshape([grid], [1, 1, 2])
	pred_wh_half = 0.5 * pred_wh_cell
	pred_xy_center = tf.math.add(pred_xy_tensor, offset) * 32.
	pred_mins = pred_xy_center - pred_wh_half
	pred_maxes = pred_xy_center + pred_wh_half

	# adjust intersection
	intersect_min = tf.maximum(pred_mins, true_mins)
	intersect_max = tf.minimum(pred_maxes, true_maxes)
	intersect_wh = tf.maximum(intersect_max - intersect_min, 0.0)
	intersect_areas = intersect_wh[:, :, 0] * intersect_wh[:, :, 1]

	# adjust areas
	true_areas = true_wh_cell[:, :, 0] * true_wh_cell[:, :, 1]
	pred_areas = pred_wh_cell[:, :, 0] * pred_wh_cell[:, :, 1]
	union_areas = true_areas + pred_areas - intersect_areas
	iou_scores  = tf.truediv(intersect_areas, union_areas) # shape(None, 40)

	# when there's no object in cell iou should be zero
	iou_scores_corrected = y_true[:, :, 0] * iou_scores

	# confidence error: pred_conf - iou
	conf_error = pred_conf_tensor - iou_scores_corrected

	# compute error
	conf_mask = K.expand_dims(y_true[:, :, 0], axis=-1)
	coord_mask = K.concatenate([conf_mask, conf_mask], axis=-1)

	# location error
	batch_size = 64.0
	loss_xy = tf.reduce_sum(tf.square(true_xy_tensor - pred_xy_tensor) * coord_mask) / batch_size

	# size error
	loss_wh = tf.reduce_sum(tf.square(tf.sqrt(true_wh_tensor + K.epsilon()) - tf.sqrt(pred_wh_tensor + K.epsilon())) * coord_mask) / batch_size

	# depth error
	loss_mean = tf.reduce_sum(tf.square(true_m_tensor - pred_m_tensor) * conf_mask) / batch_size

	# variance depth error
	loss_var = tf.reduce_sum(tf.square(true_v_tensor - pred_v_tensor) * conf_mask) / batch_size

	# confidence object error
	loss_object = tf.reduce_sum(K.expand_dims(tf.square(conf_error), axis=-1) * conf_mask) / batch_size

	# confidence non object error
	loss_non_object = tf.reduce_sum(K.expand_dims(tf.square(conf_error), axis=-1) * (1. - conf_mask)) / batch_size

	# total loss
	loss = 1.0 * loss_object + 0.001 * loss_non_object + 2.5 * loss_xy + 2.5 * loss_wh + 1.0 * loss_mean + 1.0 * loss_var

	return loss

def yolo_objconf_loss(y_true, y_pred):
	# adjust confidence
	pred_conf_tensor = K.sigmoid(y_pred[..., 0])

	# Compute IOU
	iou_scores  = iou(y_true, y_pred) # shape(None, 40)

	# when there's no object in cell iou should be zero
	iou_scores_corrected = y_true[..., 0] * iou_scores

	# confidence error: pred_conf - iou
	conf_error = pred_conf_tensor - iou_scores_corrected

	# compute error
	conf_mask = K.expand_dims(y_true[..., 0], axis=-1)

	# location error
	batch_size = 64.0

	# confidence object error
	loss_object = tf.reduce_sum(K.expand_dims(tf.square(conf_error), axis=-1) * conf_mask) / batch_size

	# total loss
	loss = 1.0 * loss_object

	return loss

def yolo_nonobjconf_loss(y_true, y_pred):
	# adjust confidence
	pred_conf_tensor = K.sigmoid(y_pred[..., 0])

	# Compute IOU
	iou_scores  = iou(y_true, y_pred) # shape(None, 40)

	# when there's no object in cell iou should be zero
	iou_scores_corrected = y_true[..., 0] * iou_scores

	# confidence error: pred_conf - iou
	conf_error = pred_conf_tensor - iou_scores_corrected

	# compute error
	conf_mask = K.expand_dims(y_true[..., 0], axis=-1)

	# location error
	batch_size = 64.0

	# confidence non object error
	loss_non_object = tf.reduce_sum(K.expand_dims(tf.square(conf_error), axis=-1) * (1. - conf_mask)) / batch_size

	# total loss
	loss = 0.001 * loss_non_object

	return loss

def yolo_xy_loss(y_true, y_pred):
	batch_size = 64.0
	conf_mask = K.expand_dims(y_true[..., 0], axis=-1)
	coord_mask = K.concatenate([conf_mask, conf_mask], axis=-1)
	# adjust x, y
	true_xy_tensor = y_true[..., 1:3] # relative position to the containing cell
	pred_xy_tensor = K.sigmoid(y_pred[..., 1:3]) # relative to position to the containing cell
	loss_xy = tf.reduce_sum(tf.square(true_xy_tensor - pred_xy_tensor) * coord_mask) / batch_size
	return 2.5 * loss_xy

def yolo_wh_loss(y_true, y_pred):
	size = (tf.shape(y_true)).shape
	batch_size = 64.0
	conf_mask = K.expand_dims(y_true[..., 0], axis=-1)
	coord_mask = K.concatenate([conf_mask, conf_mask], axis=-1)
	# adjust w, h
	true_wh_tensor = y_true[..., 3:5] # relative to image shape
	if size == 5:
		# Anchors
		anchors = np.array([[0.34755122, 0.84069513],
							[0.14585618, 0.25650666]], dtype=np.float32)
		pred_wh_tensor = K.exp(y_pred[..., 3:5]) * K.reshape(anchors, [1, 1, 1, 2, 2]) # relative to image shape
	else:
		pred_wh_tensor = K.sigmoid(y_pred[..., 3:5]) # relative to image shape
	loss_wh = tf.reduce_sum(tf.square(tf.sqrt(true_wh_tensor + K.epsilon()) - tf.sqrt(pred_wh_tensor + K.epsilon())) * coord_mask) / batch_size
	return 2.5 * loss_wh

def yolo_mean_loss(y_true, y_pred):
	batch_size = 64.0
	conf_mask = K.expand_dims(y_true[..., 0], axis=-1)
	# adjust mean
	true_m_tensor = K.expand_dims(y_true[..., 5], axis=-1)
	pred_m_tensor = K.expand_dims(y_pred[..., 5], axis=-1)
	loss_mean = tf.reduce_sum(tf.square(true_m_tensor - pred_m_tensor) * conf_mask) / batch_size
	return 1.0 * loss_mean

def yolo_var_loss(y_true, y_pred):
	batch_size = 64.0
	conf_mask = K.expand_dims(y_true[..., 0], axis=-1)
	# adjust mean
	true_v_tensor = K.expand_dims(y_true[..., 6], axis=-1)
	pred_v_tensor = K.expand_dims(y_pred[..., 6], axis=-1)
	loss_var = tf.reduce_sum(tf.square(true_v_tensor - pred_v_tensor) * conf_mask) / batch_size
	return 1.0 * loss_var

def yolo_v2_loss(y_true, y_pred):
	# Adjust prediction
	# Anchors
	anchors = np.array([[0.34755122, 0.84069513],
						[0.14585618, 0.25650666]], dtype=np.float32)
	# adjust x, y
	pred_xy_tensor = K.sigmoid(y_pred[:, :, :, :, 1:3]) # relative to position to the containing cell

	# adjust w, h
	pred_wh_tensor = K.exp(y_pred[:, :, :, :, 3:5]) * K.reshape(anchors, [1, 1, 1, 2, 2]) # relative to image shape

	# adjust confidence
	pred_conf_tensor = K.sigmoid(y_pred[:, :, :, :, 0])

	# adjust mean
	#pred_m_tensor = K.expand_dims(y_pred[:, :, :, :, 5], axis=-1)

	# adjust variance
	#pred_v_tensor = K.expand_dims(y_pred[:, :, :, :, 6], axis=-1)

	# Adjust true tensor
	# adjust x, y
	true_xy_tensor = y_true[:, :, :, :, 1:3] # relative position to the containing cell

	# adjust w, h
	true_wh_tensor = y_true[:, :, :, :, 3:5] # relative to image shape

	# adjust mean
	#true_m_tensor = K.expand_dims(y_true[:, :, :, :, 5], axis=-1)

	# adjust variance
	#true_v_tensor = K.expand_dims(y_true[:, :, :, :, 6], axis=-1)

	# Compute confidence
	# grid dimentions: x, y
	grid = [256., 160.]
	rows = []
	cols = []
	for y in range(5):
		for x in range(8):
			cell = [x, y]
			offset = [cell, cell]
			cols.append(offset)
		rows.append(cols)
		cols = []
	offset = K.reshape(tf.convert_to_tensor(rows, dtype=np.float32), [1, 5, 8, 2, 2])

	# adjust true min and max
	true_wh_cell = true_wh_tensor * K.reshape([grid, grid], [1, 1, 1, 2, 2])
	true_wh_half = 0.5 * true_wh_cell
	true_xy_center = tf.math.add(true_xy_tensor, offset) * 32.
	true_mins = true_xy_center - true_wh_half
	true_maxes = true_xy_center + true_wh_half

	# adjust pred min and max
	pred_wh_cell = pred_wh_tensor * K.reshape([grid, grid], [1, 1, 1, 2, 2])
	pred_wh_half = 0.5 * pred_wh_cell
	pred_xy_center = tf.math.add(pred_xy_tensor, offset) * 32.
	pred_mins = pred_xy_center - pred_wh_half
	pred_maxes = pred_xy_center + pred_wh_half

	# adjust intersection
	intersect_min = tf.maximum(pred_mins, true_mins)
	intersect_max = tf.minimum(pred_maxes, true_maxes)
	intersect_wh = tf.maximum(intersect_max - intersect_min, 0.0)
	intersect_areas = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

	# adjust areas
	true_areas = true_wh_cell[:, :, :, :, 0] * true_wh_cell[:, :, :, :, 1]
	pred_areas = pred_wh_cell[:, :, :, :, 0] * pred_wh_cell[:, :, :, :, 1]
	union_areas = true_areas + pred_areas - intersect_areas
	iou_scores  = tf.truediv(intersect_areas, union_areas) # shape(None, 5, 8, 2)

	# when there is no object in cell iou should be zero
	iou_scores_corrected = y_true[:, :, :, :, 0] * iou_scores

	# confidence error: pred_conf - iou
	conf_error = pred_conf_tensor - iou_scores_corrected

	# compute error
	conf_mask = K.expand_dims(y_true[:, :, :, :, 0], axis=-1)
	coord_mask = K.concatenate([conf_mask, conf_mask], axis=-1)

	# location error
	batch_size = 64.0
	loss_xy = tf.reduce_sum(tf.square(true_xy_tensor - pred_xy_tensor) * coord_mask) / batch_size

	# size error
	loss_wh = tf.reduce_sum(tf.square(tf.sqrt(true_wh_tensor + K.epsilon()) - tf.sqrt(pred_wh_tensor + K.epsilon())) * coord_mask) / batch_size

	# depth error
	#loss_mean = tf.reduce_sum(tf.square(true_m_tensor - pred_m_tensor) * conf_mask) / batch_size

	# variance depth error
	#loss_var = tf.reduce_sum(tf.square(true_v_tensor - pred_v_tensor) * conf_mask) / batch_size

	# confidence object error
	loss_object = tf.reduce_sum(K.expand_dims(tf.square(conf_error), axis=-1) * conf_mask) / batch_size

	# confidence non object error
	loss_non_object = tf.reduce_sum(K.expand_dims(tf.square(conf_error), axis=-1) * (1. - conf_mask)) / batch_size

	# total loss
	loss = 1.0 * loss_object + 0.005 * loss_non_object + 3.5 * loss_xy + 3.5 * loss_wh #+ 1.0 * loss_mean + 1.0 * loss_var

	return loss

def yolo_v3_loss(y_true, y_pred):
	# Adjust prediction
	# Anchors
	anchors = np.array([[0.34755122, 0.84069513],
						[0.14585618, 0.25650666]], dtype=np.float32)
	# adjust x, y
	pred_xy_tensor = K.sigmoid(y_pred[:, :, :, :, 1:3]) # relative to position to the containing cell

	# adjust w, h
	pred_wh_tensor = K.exp(y_pred[:, :, :, :, 3:5]) * K.reshape(anchors, [1, 1, 1, 2, 2]) # relative to image shape

	# adjust confidence
	pred_conf_tensor = K.sigmoid(y_pred[:, :, :, :, 0])

	# adjust mean
	pred_m_tensor = K.expand_dims(y_pred[:, :, :, :, 5], axis=-1)

	# adjust variance
	pred_v_tensor = K.expand_dims(y_pred[:, :, :, :, 6], axis=-1)

	# Adjust true tensor
	# adjust x, y
	true_xy_tensor = y_true[:, :, :, :, 1:3] # relative position to the containing cell

	# adjust w, h
	true_wh_tensor = y_true[:, :, :, :, 3:5] # relative to image shape

	# adjust mean
	true_m_tensor = K.expand_dims(y_true[:, :, :, :, 5], axis=-1)

	# adjust variance
	true_v_tensor = K.expand_dims(y_true[:, :, :, :, 6], axis=-1)

	# Compute confidence
	max_mean_depth = tf.maximum(pred_m_tensor, true_m_tensor)
	min_mean_depth = tf.minimum(pred_m_tensor, true_m_tensor)
	overlap_scores = tf.truediv(min_mean_depth, max_mean_depth)

	# when there is no object in cell iou should be zero
	overlap_scores_corrected = y_true[:, :, :, :, 0] * overlap_scores

	# confidence error: pred_conf - iou
	conf_error = pred_conf_tensor - overlap_scores_corrected

	# compute error
	conf_mask = K.expand_dims(y_true[:, :, :, :, 0], axis=-1)
	coord_mask = K.concatenate([conf_mask, conf_mask], axis=-1)

	# location error
	batch_size = 64.0
	loss_xy = tf.reduce_sum(tf.square(true_xy_tensor - pred_xy_tensor) * coord_mask) / batch_size

	# size error
	loss_wh = tf.reduce_sum(tf.square(tf.sqrt(true_wh_tensor + K.epsilon()) - tf.sqrt(pred_wh_tensor + K.epsilon())) * coord_mask) / batch_size

	# depth error
	loss_mean = tf.reduce_sum(tf.square(true_m_tensor - pred_m_tensor) * conf_mask) / batch_size

	# variance depth error
	loss_var = tf.reduce_sum(tf.square(true_v_tensor - pred_v_tensor) * conf_mask) / batch_size

	# confidence object error
	loss_object = tf.reduce_sum(K.expand_dims(tf.square(conf_error), axis=-1) * conf_mask) / batch_size

	# confidence non object error
	loss_non_object = tf.reduce_sum(K.expand_dims(tf.square(conf_error), axis=-1) * (1. - conf_mask)) / batch_size

	# total loss
	loss = 1.0 * loss_object + 0.005 * loss_non_object + 3.5 * loss_xy + 3.5 * loss_wh + 1.0 * loss_mean + 1.0 * loss_var

	return loss
