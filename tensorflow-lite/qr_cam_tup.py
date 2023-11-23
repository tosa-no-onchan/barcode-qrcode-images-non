# -*- coding: utf-8 -*-
'''
qr_cam_tpu.py
'''

USE_TF=False

import sys
import time

import cv2
import numpy as np
from PIL import Image

if USE_TF == True:
  import tensorflow as tf
import tflite_runtime.interpreter as tflite
from pycoral.adapters import common

import platform
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]
#EDGETPU_SHARED_LIB = "libedgetpu.so.1"

import deqr

from utils import letterbox_image

#model_path = 'model.tflite'
model_path = 'tpu/model_edgetpu.tflite'

# Load the labels into a list
classes = ['QR_CODE']

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

'''
  preprocess_image_cam(img, size):
    img :
    input_size : (width , height)
 update by nishi 2021.2.25
'''
def preprocess_image_cam(img, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  #img = cv2.imread(image_path)
  # 左右反転
  #img = cv2.flip(img, 1)

  img_tf,v_size = letterbox_image(img, input_size,func=0)

  # Convert the image from BGR to RGB as required by the TFLite model.
  #rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  if False:
    # 画像の表示
    cv2.imshow("Image", img_tf)
    # キー入力待ち(ここで画像が表示される)
    cv2.waitKey()

  original_image = img

  #img_tf=img_tf[np.newaxis,:, :,:]
  resized_img =img_tf[np.newaxis,:, :,:]

  return resized_img, original_image,v_size

#--------------------
# https://www.tensorflow.org/lite/api_docs/python/tf/lite/Interpreter
# https://www.tensorflow.org/lite/guide/python?hl=ja
# https://www.tensorflow.org/lite/guide/inference?hl=ja
#
# reffer with the following.
# ~/kivy_3.9/lib/python3.9/site-package/pycoral/adaapters/detect.py
#   line 184 def get_objects(()
#--------------------
def detect_objects_tpu(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  #signature_fn = interpreter.get_signature_runner()

  # Feed the input image to the model
  #output = signature_fn(images=image)

  input_details = interpreter.get_input_details()
  #output_details = interpreter.get_output_details()
  #input_shape = input_details[0]["shape"]

  interpreter.set_tensor(input_details[0]['index'], image)

  # Run model
  interpreter.invoke()

  signature_list = interpreter._get_full_signature_list()

  # pylint: enable=protected-access
  if signature_list:
    if len(signature_list) > 1:
      raise ValueError('Only support model with one signature.')
    signature = signature_list[next(iter(signature_list))]
    count = int(interpreter.tensor(signature['outputs']['output_0'])()[0])
    scores = interpreter.tensor(signature['outputs']['output_1'])()[0]
    class_ids = interpreter.tensor(signature['outputs']['output_2'])()[0]
    boxes = interpreter.tensor(signature['outputs']['output_3'])()[0]
  elif common.output_tensor(interpreter, 3).size == 1:
    #print('detect_objects_tpu: #4')
    boxes = common.output_tensor(interpreter, 0)[0]
    class_ids = common.output_tensor(interpreter, 1)[0]
    scores = common.output_tensor(interpreter, 2)[0]
    count = int(common.output_tensor(interpreter, 3)[0])
  else:
    #print('detect_objects_tpu: #5')
    scores = common.output_tensor(interpreter, 0)[0]
    boxes = common.output_tensor(interpreter, 1)[0]
    count = (int)(common.output_tensor(interpreter, 2)[0])
    class_ids = common.output_tensor(interpreter, 3)[0]

  if False:
    # Get all outputs from the model
    #print('type(output):',type(output))
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    class_ids = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

  #print('count:',count)   # count: 25
  results = []
  if True:
    for i in range(count):
      if scores[i] >= threshold:
        result = {
          'bounding_box': boxes[i],
          'class_id': class_ids[i],
          'score': scores[i]
        }
        results.append(result)
  return results

#--------------------
# https://www.tensorflow.org/lite/api_docs/python/tf/lite/Interpreter
# https://www.tensorflow.org/lite/guide/python?hl=ja
# https://www.tensorflow.org/lite/guide/inference?hl=ja
#--------------------
def detect_objects(signature_fn, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  #signature_fn = interpreter.get_signature_runner()

  # Feed the input image to the model
  output = signature_fn(images=image)

  # Get all outputs from the model
  #print('type(output):',type(output))
  count = int(np.squeeze(output['output_0']))
  scores = np.squeeze(output['output_1'])
  classes = np.squeeze(output['output_2'])
  boxes = np.squeeze(output['output_3'])

  #print('count:',count)
  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results

'''
  run_odt_cam(img, interpreter,input_size):
    img :
    interpreter : 
    input_size : (width , height)
    threshold :
'''
def run_odt_cam(img, interpreter,input_size, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  #_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  input_width,input_height = input_size

  decoder = deqr.QRdecDecoder()
  #decoder = deqr.QuircDecoder()

  # Load the input image and preprocess it
  preprocessed_image, original_image,v_size = preprocess_image_cam(
      img,
      (input_width,input_height)
    )

  # Run object detection on the input image
  results = detect_objects_tpu(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  if USE_TF==True:
    original_image_np = original_image.numpy().astype(np.uint8)
  else:
    original_image_np = original_image

  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    #print(xmin,xmax)
    if False:
      xmin = int(xmin * original_image_np.shape[1])
      xmax = int(xmax * original_image_np.shape[1])
      ymin = int(ymin * original_image_np.shape[0])
      ymax = int(ymax * original_image_np.shape[0])
    else:
      w_off = int((v_size[0] - original_image_np.shape[0]) /2) 
      h_off = int((v_size[1] - original_image_np.shape[1]) /2) 
      xmin = int(xmin * v_size[1]) - h_off
      xmax = int(xmax * v_size[1]) - h_off
      ymin = int(ymin * v_size[0]) - w_off
      ymax = int(ymax * v_size[0]) - w_off

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # img[top : bottom, left : right]
    # サンプル1の切り出し、保存
    #img1 = img[0 : 50, 0: 50]
    img_qr = original_image_np[ymin:ymax,xmin:xmax]
    #print('img_qr.shape',img_qr.shape)
    data=''
    if img_qr.shape[0] > 9 and img_qr.shape[1] >9:
      if False:
        cv2.imshow("Image", img_qr)
        # キー入力待ち(ここで画像が表示される)
        cv2.waitKey()
      decoded_codes = decoder.decode(img_qr)
      #print('len(decoded_codes):',len(decoded_codes))
      if len(decoded_codes) > 0:
        print('img_qr.shape',img_qr.shape)  # best box size is img_qr.shape (270, 260, 3)
      for code in decoded_codes:
        #print('type(code):',type(code)) # type(item): <class 'deqr.datatypes.QRCode'>
        #print(code.data_entries)    # (QrCodeData(type=QRDataType.BYTE, data=VERSION 2 8CM),)
        qrCodeData=code.data_entries[0] 
        #print(qrCodeData)           # QrCodeData(type=QRDataType.BYTE, data=VERSION 2 8CM)
        data = qrCodeData.data
        print('data:',data)

    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    if data != '':
      label=data
    else:
      label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8


DETECTION_THRESHOLD = 0.3

if __name__ == '__main__':
  # Load the TFLite model
  if False:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
  else:
    #interpreter = tflite.Interpreter(model_path=model_path)

    model_file, *device = model_path.split('@')
    interpreter = tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                {'device': device[0]} if device else {})
        ])
    print('model_file:',model_file)

    interpreter.allocate_tensors()

  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  #signature_fn = interpreter.get_signature_runner()

  camera_id=0
  width=640
  height=480

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    counter += 1
    image = cv2.flip(image, 1)

    detection_result_image = run_odt_cam(image, interpreter,(input_width,input_height),threshold=DETECTION_THRESHOLD)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(detection_result_image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector',detection_result_image)

  cap.release()
  cv2.destroyAllWindows()


