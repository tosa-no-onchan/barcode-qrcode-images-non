# -*- coding: utf-8 -*-
'''
test_cv_tpu.py

https://torque.github.io/deqr-docs/latest-dev/getting-started.html

'''

USE_TF=False

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


def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = cv2.imread(image_path)

  # 左右反転
  #img = cv2.flip(img, 1)

  img_tf,v_size = letterbox_image(img, input_size,func=0)

  # Convert the image from BGR to RGB as required by the TFLite model.
  #rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  if True:
    # 画像の表示
    cv2.imshow("Image", img_tf)

    # キー入力待ち(ここで画像が表示される)
    cv2.waitKey()

  #print('img.shape:',img.shape) # (288, 512, 3)
  #print("img.dtype=",img.dtype) # uint8

  #print('img_tf.shape:',img_tf.shape) # (288, 512, 3)
  #print("img_tf.dtype=",img_tf.dtype) # uint8

  original_image = img

  img_tf=img_tf[np.newaxis,:, :,:]
  resized_img =img_tf

  #print('2.img_tf.shape:', img_tf.shape) # (288, 512, 3)
  #print("2.img_tf.dtype=",img_tf.dtype) # uint8
  #print("2.type(img_tf)=",type(img_tf)) # uint8

  if USE_TF==True:
    scale=1.0
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)

    resized_img_np = resized_img.numpy().astype(np.uint8)
    #print('resized_img_np.shape:',resized_img_np.shape)

    if True:
      cv2.imshow("Image", resized_img_np[0])
      # キー入力待ち(ここで画像が表示される)
      cv2.waitKey()

  #print('3.resized_img.shape:',resized_img.shape) # (288, 512, 3)
  #print("3.resized_img.dtype=",resized_img.dtype) # uint8
  #print("3.type(resized_img)=",type(resized_img)) # uint8

  return resized_img, original_image,v_size

def preprocess_image_org(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  resized_img = tf.cast(resized_img, dtype=tf.uint8)
  return resized_img, original_image

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
    print('detect_objects_tpu: #4')
    boxes = common.output_tensor(interpreter, 0)[0]
    class_ids = common.output_tensor(interpreter, 1)[0]
    scores = common.output_tensor(interpreter, 2)[0]
    count = int(common.output_tensor(interpreter, 3)[0])
  else:
    print('detect_objects_tpu: #5')
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

  print('count:',count)   # count: 25

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

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  signature_fn = interpreter.get_signature_runner()

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


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  print('input_height:',input_height,' input_width:', input_width)

  decoder = deqr.QRdecDecoder()
  #decoder = deqr.QuircDecoder()

  # Load the input image and preprocess it
  preprocessed_image, original_image,v_size = preprocess_image(
      image_path,
      (input_height, input_width)
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
    data=''
    if img_qr.shape[0] > 9 and img_qr.shape[1] > 9:
      if True:
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

TEMP_FILE = 'test.png'
TEMP_FILE ='test/rotationsimage022.jpg'
#TEMP_FILE = 'test/curvedimage021.jpg'
#TEMP_FILE ='test/curvedimage005.jpg'

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

  # Run inference and draw detection result on the local copy of the original file
  detection_result_image = run_odt_and_draw_results(
      TEMP_FILE,
      interpreter,
      threshold=DETECTION_THRESHOLD
  )

  # Show the detection result
  image  = Image.fromarray(detection_result_image)
  image.show()