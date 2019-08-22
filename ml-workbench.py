import h5py
from nms import nms
from NPDScan import NPDScan
from imutils.video import FPS
from imutils.video import VideoStream
import imutils
import dlib
import numpy as np
import math
import sys
import cv2
import os
import threading
import time
from pprint import pprint
import json
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk as gtk, GObject, Gdk, GdkPixbuf, GLib, Gtk


GLib.threads_init()
GObject.threads_init()
Gdk.threads_init()

mymutex = threading.Lock()
dimg1 = GdkPixbuf.Pixbuf.new_from_file('./images/media-player-128.png')
dimg2 = GdkPixbuf.Pixbuf.new_from_file('./images/media-player-128.png')

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

video_store = './video/'
frames_per_second = 24.0
res = '480p'
camrelease = False
streamvideo = True

def VideoPlayer(filename, drawing_area, player, e, play, pause):
  if player == 1:
    global dimg1, dimg_available1
  else:
    global dimg2, dimg_available2

  cap = cv2.VideoCapture(filename)
  # cap = cv2.QueryFrame(cap)
  while(cap.isOpened()):
    if play1:
      mymutex.acquire()
      ret, img = cap.read()
      if img is not None:
        boxAllocation = drawing_area.get_allocation()
        img = cv2.resize(img, (boxAllocation.width,
                                     boxAllocation.height))

        # opencv by default load BGR colorspace. Gtk supports RGB hance the conversion
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if player == 1:
          dimg1 = GdkPixbuf.Pixbuf.new_from_data(img.tostring(),
                                                GdkPixbuf.Colorspace.RGB, False, 8,
                                                img.shape[1],
                                                img.shape[0],
                                                img.shape[2]*img.shape[1], None, None)
        else:
          dimg2 = GdkPixbuf.Pixbuf.new_from_data(img.tostring(),
                                                GdkPixbuf.Colorspace.RGB, False, 8,
                                                img.shape[1],
                                                img.shape[0],
                                                img.shape[2]*img.shape[1], None, None)
        drawing_area.queue_draw()
        mymutex.release()
        time.sleep(0.03)
      else:
        pause.set_sensitive(False)
        play.set_sensitive(True)
        mymutex.release()
        break

def VideoPlayerWebcam(drawing_area, taskcombobox, modelcombobox, player, ):
  # global dimg1, dimg_available1, camrelease, dimg2, dimg_available2

  # cap = cv2.VideoCapture("./video/face_detection_Haar cascade_head-pose-face-detection-female_5.avi")
  cap = cv2.VideoCapture(0)
  with open('./config.json') as f:
    data = json.load(f)

  while(cap.isOpened()):
    if play1:
      model = ""
      try:
        task = taskcombobox.get_model()[taskcombobox.get_active()][0]
        model = modelcombobox.get_model()[modelcombobox.get_active()][0]
      except Exception as e:
        print(e)
      
      for c in data['tasks']:
        if c["task_name"] == task:
          for idx, c1 in enumerate(c["model"]):
                
            if c1["model_name"] == model :
              modelfile = c1["model_path"]
              modelconfig = c1["model_config_path"]

      mymutex.acquire()
      ret, frame = cap.read()
      if frame is not None:
        if task == "face_detection" and model == "Haar cascade":
          face_cascade = cv2.CascadeClassifier(modelfile)
          frame = FaceDetection().haarCascade(model, frame, face_cascade)
        elif (task == "face_detection" and (model == "MTCNN" or model == "Resnet SSD")):
          filename, ext = os.path.splitext(modelfile)
          if ext == ".caffemodel":
            net = cv2.dnn.readNetFromCaffe(modelconfig, modelfile)
          else:
            net = cv2.dnn.readNetFromTensorflow(modelconfig, modelfile)
          frame = FaceDetection().caffeeAndTensorModel(model, frame, net, FPS().start())
        elif (task == "face_detection" and (model == "HOG dlib" or model == "MMOD lib")):
          if model == "HOG dlib":
            detector = dlib.get_frontal_face_detector()
          else:
            detector = dlib.cnn_face_detection_model_v1(modelfile)
          frame1 = FaceDetection().dlib(model, frame, detector)
        elif task == "face_detection" and model == "NPD":
          minFace = 20
          maxFace = 4000
          overlap = 0.5
          f = h5py.File(model1file, 'r')
          npdModel = {n: np.array(v) for n, v in f.get('npdModel').items()}
          frame = FaceDetection().npd(model, frame, npdModel, minFace, maxFace, overlap)

        webcamframe(drawing_area, frame, player)
        mymutex.release()
        if camrelease:
           break
      else:
        mymutex.release()
        break
  cap.release()

# def VideoPlayerWebcam(drawing_area1, drawing_area2, taskcombobox, model1combobox, model2combobox):
#   # global dimg1, dimg_available1, camrelease, dimg2, dimg_available2

#   cap = cv2.VideoCapture(0)
#   with open('./config.json') as f:
#     data = json.load(f)

#   while(cap.isOpened()):
#     if play1:
#       model = ""
#       try:
#         task = taskcombobox.get_model()[taskcombobox.get_active()][0]
#         model1 = model1combobox.get_model()[model1combobox.get_active()][0]
#         model2 = model2combobox.get_model()[model2combobox.get_active()][0]
#       except Exception as e:
#         print(e)
      
#       for c in data['tasks']:
#         if c["task_name"] == task:
#           for idx, c1 in enumerate(c["model"]):
                
#             if c1["model_name"] == model1 :
#               model1file = c1["model_path"]
#               model1config = c1["model_config_path"]

#             if c1["model_name"] == model2 :
#               model2file = c1["model_path"]
#               model2config = c1["model_config_path"]

#       mymutex.acquire()
#       ret, frame = cap.read()
#       if frame is not None:
#         frame1 = frame.copy()
#         frame2 = frame.copy()
#         if task == "face_detection" and model1 == "Haar cascade":
#           face_cascade = cv2.CascadeClassifier(model1file)
#           frame1 = FaceDetection().haarCascade(model1, frame1, face_cascade)
#         elif (task == "face_detection" and (model1 == "MTCNN" or model1 == "Resnet SSD")):
#           filename, ext = os.path.splitext(model1file)
#           if ext == ".caffemodel":
#             net = cv2.dnn.readNetFromCaffe(model1config, model1file)
#           else:
#             net = cv2.dnn.readNetFromTensorflow(model1config, model1file)
#           frame1 = FaceDetection().caffeeAndTensorModel(model1, frame1, net, FPS().start())
#         elif (task == "face_detection" and (model1 == "HOG dlib" or model1 == "MMOD lib")):
#           if model1 == "HOG dlib":
#             detector = dlib.get_frontal_face_detector()
#           else:
#             detector = dlib.cnn_face_detection_model_v1(model1file)
#           frame1 = FaceDetection().dlib(model1, frame1, detector)
#         elif task == "face_detection" and model1 == "NPD":
#           minFace = 20
#           maxFace = 4000
#           overlap = 0.5
#           f = h5py.File(model1file, 'r')
#           npdModel = {n: np.array(v) for n, v in f.get('npdModel').items()}
#           frame1 = FaceDetection().npd(model1, frame1, npdModel, minFace, maxFace, overlap)

#         webcamframe(drawing_area1, frame1, 1)

#         if task == "face_detection" and model2 == "Haar cascade":
#           face_cascade = cv2.CascadeClassifier(model2file)
#           frame2 = FaceDetection().haarCascade(model2, frame2, face_cascade)
#         elif (task == "face_detection" and (model2 == "MTCNN" or model2 == "Resnet SSD")):
#           filename, ext = os.path.splitext(model2file)
#           if ext == ".caffemodel":
#             net = cv2.dnn.readNetFromCaffe(model2config, model2file)
#           else:
#             net = cv2.dnn.readNetFromTensorflow(model2config, model2file)
#           frame2 = FaceDetection().caffeeAndTensorModel(model2, frame2, net, FPS().start())
#         elif (task == "face_detection" and (model2 == "HOG dlib" or model2 == "MMOD lib")):
#           if model2 == "HOG dlib":
#             detector = dlib.get_frontal_face_detector()
#           else:
#             detector = dlib.cnn_face_detection_model_v1(model2file)
#           frame2 = FaceDetection().dlib(model2, frame2, detector)
#         elif task == "face_detection" and model2 == "NPD":
#           minFace = 20
#           maxFace = 4000
#           overlap = 0.5
#           f = h5py.File(model2file, 'r')
#           npdModel = {n: np.array(v) for n, v in f.get('npdModel').items()}
#           frame2 = FaceDetection().npd(model2, frame2, npdModel, minFace, maxFace, overlap)
#         webcamframe(drawing_area2, frame2, 2)
#         mymutex.release()
#         time.sleep(0.03)
#         if camrelease:
#            break
#       else:
#         mymutex.release()
#         break
#   # cap.release()

def webcamframe(drawing_area, frame, player):
  global dimg1, dimg_available1, camrelease, dimg2, dimg_available2

  boxAllocation = drawing_area.get_allocation()
        
  frame = cv2.resize(frame, (boxAllocation.width,
                               boxAllocation.height))
  
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  if player == 1:
    dimg1 = GdkPixbuf.Pixbuf.new_from_data(frame.tostring(),
                                          GdkPixbuf.Colorspace.RGB, False, 8,
                                          frame.shape[1],
                                          frame.shape[0],
                                          frame.shape[2]*frame.shape[1], None, None)
  else:
    dimg2 = GdkPixbuf.Pixbuf.new_from_data(frame.tostring(),
                                          GdkPixbuf.Colorspace.RGB, False, 8,
                                          frame.shape[1],
                                          frame.shape[0],
                                          frame.shape[2]*frame.shape[1], None, None)

  drawing_area.queue_draw()



class DetectionFile:
  def __init__(self, task, model, videopath, model_path, model_config_path, progressfunc, modelid, modelprocesstime, framerate):
    self.videopath = videopath
    self.model_path = model_path
    self.model_config_path = model_config_path
    self.progressfunc = progressfunc
    self.modelprocesstime = modelprocesstime
    self.task = task
    self.model = model
    self.modelid = modelid
    self.cap = cv2.VideoCapture(self.videopath)
    self.progess = 0
    self.fileaddress = video_store + task + "_" + model + "_" + os.path.split(videopath)[1].split('.')[0] + "_" + str(framerate) + ".avi"
    self.fps = FPS().start()
    self.framerate = framerate
    self.framecount = framerate
    self.frameloop = True
    self.out = cv2.VideoWriter(self.fileaddress, self.get_video_type(
        self.videopath), frames_per_second, (int(self.cap.get(3)), int(self.cap.get(4))))

  def change_res(self, cap, width, height):
    self.cap.set(3, width)
    self.cap.set(4, height)

  def get_video_type(self, filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

  def get_dims(self, cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    self.change_res(self.cap, width, height)
    return width, height
  
  def get_video_ext(self, file):
    filename, ext = os.path.splitext(file)
    return ext

  def example_target(self):

    if int(major_ver) < 3:
      fps = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
      print(
          "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
      fps = self.cap.get(cv2.CAP_PROP_FPS)
      print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {}".format(fps))

    length = int(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self.framerate)
    starttime = time.time()
    if self.cap.isOpened():
      while(self.cap.isOpened()):
        ret, frame = self.cap.read(self.framecount)
        if self.framecount <= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
          self.framecount += self.framerate
          if self.task == "face_detection" and self.model == "Haar cascade":
            face_cascade = cv2.CascadeClassifier(self.model_path)
            # eye_cascade = cv2.CascadeClassifier(self.model_config_path)
            frame = FaceDetection().haarCascade(self.model, frame, face_cascade)
          elif (self.task == "face_detection" and (self.model == "MTCNN" or self.model == "Resnet SSD")):
            if self.get_video_ext(self.model_path) == ".caffemodel":
                  net = cv2.dnn.readNetFromCaffe(
                  self.model_config_path, self.model_path)
            else:
              net = cv2.dnn.readNetFromTensorflow(
                  self.model_config_path, self.model_path)
            frame = FaceDetection().caffeeAndTensorModel(self.model, frame, net, self.fps)
          elif (self.task == "face_detection" and (self.model == "HOG dlib" or self.model == "MMOD lib")):
            if self.model == "HOG dlib":
              detector = dlib.get_frontal_face_detector()
            else:
              detector = dlib.cnn_face_detection_model_v1(self.model_path)
              
            frame = FaceDetection().dlib(self.model, frame, detector)
          elif self.task == "face_detection" and self.model == "NPD":
            minFace = 20
            maxFace = 4000
            overlap = 0.5
            f = h5py.File(self.model_path, 'r')
            npdModel = {n: np.array(v) for n, v in f.get('npdModel').items()}
            frame = FaceDetection().npd(self.model, frame, npdModel, minFace, maxFace, overlap)
          elif self.task == "face_detection" and self.model == "Pytorch":
            from detectors import FaceBoxes
            DET2 = FaceBoxes(device='cuda')
          
          self.out.write(frame)
          self.progess = self.progess + 1
          GLib.idle_add(self.progressfunc, round(
              ((self.progess/length) * 100), 1))
          # time.sleep(0.01)
        else:
          self.modelprocesstime.set_visible(True)
          self.modelprocesstime.set_text("Time : {}".format(round(time.time() - starttime, 2)));
          return False
          print("Loopend")
          break

      self.modelprocesstime.set_visible(True)
      self.modelprocesstime.set_text("Time : {}".format(round(time.time() - starttime, 2)));
      self.cap.release()
      print("Finish")
    else:
      print("Error opening video stream or file")
      dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                  Gtk.ButtonsType.OK, "Error of File")
      # dialogset_transient_for()
      dialog.format_secondary_text(
            "Error opening video stream or file")
      dialog.run()
      response = dialog.run()
      if response == Gtk.ResponseType.OK:
        print("WARN dialog closed by clicking OK button")

  def convertApply(self):
        
    thread = threading.Thread(target=self.example_target)
    thread.daemon = True
    thread.start()

class FaceDetection:
      
  def rect_to_bb(self, rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)
      
  def haarCascade(self, mode1, frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = frame[y:y+h, x:x+w]
      
      # eyes = eye_cascade.detectMultiScale(roi_gray)
      # for (ex,ey,ew,eh) in eyes:
      #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame
  
  def caffeeAndTensorModel(self, mode1, frame, net, fps):
    (origin_h, origin_w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
      confidence = detections[0, 0, i, 2]

      if confidence > 0.5:
        bounding_box = detections[0, 0, i, 3:7] * \
            np.array([origin_w, origin_h, origin_w, origin_h])
        x_start, y_start, x_end, y_end = bounding_box.astype('int')

        label = '{0:.2f}%'.format(confidence * 100)
        cv2.rectangle(frame, (x_start, y_start),
                      (x_end, y_end), (0, 0, 255), 2)
        cv2.rectangle(frame, (x_start, y_start-18),
                      (x_end, y_start), (0, 0, 255), -1)
        cv2.putText(frame, label, (x_start+2, y_start-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        fps.update()
        fps.stop()
        text = "FPS: {:.2f}".format(fps.fps())
        cv2.putText(frame, text, (15, int(origin_h * 0.92)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
  
  def dlib(self, mode1, frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
          
      if mode1 == "MMOD lib":
        (x, y, w, h) = self.rect_to_bb(rect.rect)
      else:
        (x, y, w, h) = self.rect_to_bb(rect)
      # print(i, x, y, w, h)
      # clone = frame.copy()
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      startX = x
      startY = y - 15 if y - 15 > 15 else y + 15
      cv2.putText(frame, str(i), (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame
      
  def npd(self, mode1, frame, npdModel, minFace, maxFace, overlap):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    startTime = time.time()
    rects = NPDScan(npdModel, gray, minFace, maxFace)
    print('cost:', time.time() - startTime)
    numFaces = len(rects)
    print('%s faces detected.\n' % numFaces)
    rects = nms(rects, overlap)
    if numFaces > 0:
      for rect in rects:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[3], rect[4]), (0, 255, 0), 2)
    return frame
      
    
class MLWB:
  # global model1file, model1config, model2file, model2config

  def on_main_window_destroy(self, object, data=None):
    print("Window being closed")
    gtk.main_quit()

  def on_generateResult1Button_clicked(self, object, data=None):
    if os.path.isfile(video_store + self.taskcombobox.get_model()[self.taskcombobox.get_active()][0] + "_" + self.model1combobox.get_model()[self.model1combobox.get_active()][0] + "_" + os.path.split(self.videochooserbutton.get_filename())[1].split('.')[0] + "_" + self.framerate.get_text() + ".avi"):
      dialog = Gtk.MessageDialog(self.window, 0, Gtk.MessageType.INFO,
                                  Gtk.ButtonsType.YES_NO, "The converted file exists for video")
      # dialogset_transient_for()
      dialog.format_secondary_text(
            "Please Select another video file or (model & task)")
      dialog.run()
      response = dialog.run()
      if response == Gtk.ResponseType.YES:
        self.generateresultmodel1.set_sensitive(False)
        task = self.taskcombobox.get_model()[self.taskcombobox.get_active()][0]
        model = self.model1combobox.get_model()[self.model1combobox.get_active()][0]
        
        for c in self.data['tasks']:
          if c["task_name"] == self.taskcombobox.get_model()[self.taskcombobox.get_active()][0]:
            for idx, c1 in enumerate(c["model"]):
              if c1["model_name"] == self.model1combobox.get_model()[self.model1combobox.get_active()][0]:
                detetctionfile = DetectionFile(task, model, self.videochooserbutton.get_filename(), c1["model_path"], c1["model_config_path"], self.update_progess1, "1", self.model1processtime, int(self.framerate.get_text()))
                detetctionfile.convertApply()
      elif response == Gtk.ResponseType.NO :
        self.generateresultmodel1.set_sensitive(False)
        self.generateresultmodel2.set_sensitive(True)
        print("WARN dialog closed by clicking NO button")

      dialog.destroy()
    else:
      self.generateresultmodel1.set_sensitive(False)
      task = self.taskcombobox.get_model()[self.taskcombobox.get_active()][0]
      model = self.model1combobox.get_model()[self.model1combobox.get_active()][0]
      
      for c in self.data['tasks']:
        if c["task_name"] == self.taskcombobox.get_model()[self.taskcombobox.get_active()][0]:
          for idx, c1 in enumerate(c["model"]):
            if c1["model_name"] == self.model1combobox.get_model()[self.model1combobox.get_active()][0]:
              detetctionfile = DetectionFile(task, model, self.videochooserbutton.get_filename(), c1["model_path"], c1["model_config_path"], self.update_progess1, "1", self.model1processtime, int(self.framerate.get_text()))
              detetctionfile.convertApply()

  def on_generateResult2Button_clicked(self, object, data=None):
    if os.path.isfile(video_store + self.taskcombobox.get_model()[self.taskcombobox.get_active()][0] + "_" + self.model2combobox.get_model()[self.model2combobox.get_active()][0] + "_" + os.path.split(self.videochooserbutton.get_filename())[1].split('.')[0] + "_"+ self.framerate.get_text() + ".avi"):
      dialog = Gtk.MessageDialog(self.window, 0, Gtk.MessageType.INFO,
                                  Gtk.ButtonsType.YES_NO, "The converted file exists for video")
      # dialogset_transient_for()
      dialog.format_secondary_text(
            "Please Select another video file or (model & task)")
      dialog.run()
      response = dialog.run()
      if response == Gtk.ResponseType.YES:
        self.generateresultmodel2.set_sensitive(False)
        task = self.taskcombobox.get_model()[self.taskcombobox.get_active()][0]
        model = self.model2combobox.get_model()[self.model2combobox.get_active()][0]
        for c in self.data['tasks']:
          if c["task_name"] == self.taskcombobox.get_model()[self.taskcombobox.get_active()][0]:
            for idx, c1 in enumerate(c["model"]):
              if c1["model_name"] == self.model2combobox.get_model()[self.model2combobox.get_active()][0]:
                detetctionfile = DetectionFile(task, model, self.videochooserbutton.get_filename(), c1["model_path"], c1["model_config_path"], self.update_progess2, "2", self.model2processtime, int(self.framerate.get_text()))
                detetctionfile.convertApply()
      elif response == Gtk.ResponseType.NO :
        self.generateresultmodel2.set_sensitive(False)
        self.playVideo.set_visible(True)
        self.pauseVideo.set_visible(True)
        self.pauseVideo.set_sensitive(False)
        self.playVideo.set_sensitive(True)
        print("WARN dialog closed by clicking NO button")

      dialog.destroy()
    else:
      self.generateresultmodel2.set_sensitive(False)
      task = self.taskcombobox.get_model()[self.taskcombobox.get_active()][0]
      model = self.model2combobox.get_model()[self.model2combobox.get_active()][0]
      for c in self.data['tasks']:
        if c["task_name"] == self.taskcombobox.get_model()[self.taskcombobox.get_active()][0]:
          for idx, c1 in enumerate(c["model"]):
            if c1["model_name"] == self.model2combobox.get_model()[self.model2combobox.get_active()][0]:
              detetctionfile = DetectionFile(task, model, self.videochooserbutton.get_filename(), c1["model_path"], c1["model_config_path"], self.update_progess2, "2", self.model2processtime, int(self.framerate.get_text()))
              detetctionfile.convertApply()

  def update_progess1(self, i):
    self.model1proressbar.set_fraction(i/100)
    self.model1proressbar.set_text(str(i) + " % completed")
    if i == 100:
      self.generateresultmodel2.set_sensitive(True)

    return False

  def update_progess2(self, i):
    self.model2proressbar.set_fraction(i/100)
    self.model2proressbar.set_text(str(i) + " % completed")
    if i == 100:
      self.generateresultmodel2.set_sensitive(False)
      self.playVideo.set_visible(True)
      self.pauseVideo.set_visible(True)
      self.pauseVideo.set_sensitive(False)
      self.playVideo.set_sensitive(True)

    return False

  def on_drawing_area_draw1(self,widget,cr):
    global dimg1
    mymutex.acquire()
    Gdk.cairo_set_source_pixbuf(cr, dimg1.copy(), 0, 0)
    cr.paint()
    mymutex.release()

  def on_drawing_area_draw2(self,widget,cr):
    global dimg2
    mymutex.acquire()
    Gdk.cairo_set_source_pixbuf(cr, dimg2.copy(), 0, 0)
    cr.paint()
    mymutex.release()

  def on_videoChooserButton_file_set(self, object, data=None):
    self.model2proressbar.set_fraction(0)
    self.model2proressbar.set_text(str(0) + " %")
    self.model1proressbar.set_fraction(0)
    self.model1proressbar.set_text(str(0) + " %")
    self.playVideo.set_visible(False)
    self.pauseVideo.set_visible(False)
    self.pauseVideo.set_sensitive(False)
    self.playVideo.set_sensitive(False)
    self.filename = self.videochooserbutton.get_filename()
    self.file_extension = os.path.splitext(self.filename)[1]

    if self.file_extension in ['.mp4', '.avi']:
      print("correct format")
      self.generateresultmodel1.set_sensitive(True)
    else:
      dialog = Gtk.MessageDialog(self.window, 0, Gtk.MessageType.INFO,
                                   Gtk.ButtonsType.OK, "Please Select Video File")
      # dialogset_transient_for()
      dialog.format_secondary_text(
            "The file format .mp4 and .avi")
      dialog.run()
      response = dialog.run()
      if response == Gtk.ResponseType.OK:
        print("WARN dialog closed by clicking OK button")
      dialog.destroy()
      self.generateresultmodel1.set_sensitive(False)
    # t1 = threading.Thread(target = VideoPlayer1, args=(self.filename, self.outvideo1,))
    # t1.daemon = True
    # t1.start()
    # print("video file set")

  def on_videoRadio_toggled(self, object, data=None):
    global dimg1, dimg2, camrelease
    self.videochooserbutton.set_sensitive(True)
    self.playVideo.set_visible(False)
    self.pauseVideo.set_visible(False)
    self.videochooserbutton.get_filename()
    img = cv2.imread('./images/media-player-128.png')
    dimg1 = GdkPixbuf.Pixbuf.new_from_data(img.tostring(),
                                            GdkPixbuf.Colorspace.RGB,False,8,
                                            img.shape[1],
                                            img.shape[0],
                                            img.shape[2]*img.shape[1],None,None) 
    dimg2 = GdkPixbuf.Pixbuf.new_from_data(img.tostring(),
                                            GdkPixbuf.Colorspace.RGB,False,8,
                                            img.shape[1],
                                            img.shape[0],
                                            img.shape[2]*img.shape[1],None,None) 
    self.outvideo1.queue_draw()
    self.outvideo2.queue_draw()
    camrelease = True
    try:
      if os.path.splitext(self.videochooserbutton.get_filename())[1] in ['.mp4', '.avi']:
        print("correct format")
        self.generateresultmodel1.set_sensitive(True)
        # self.generateresultmodel2.set_sensitive(True)
    except:
      print("prblem with selected file")

    print("video radio toggled")
 
  def on_webCamRadio_toggled(self, object, data=None):
    global camrelease, play1, streamvideo
    play1 = True
    self.videochooserbutton.set_sensitive(False)
    self.generateresultmodel1.set_sensitive(False)
    self.generateresultmodel2.set_sensitive(False)
    self.model2proressbar.set_fraction(0)
    self.model2proressbar.set_text(str(0) + " %")
    self.model1proressbar.set_fraction(0)
    self.model1proressbar.set_text(str(0) + " %")
    self.playVideo.set_visible(True)
    self.playVideo.set_sensitive(False)
    self.pauseVideo.set_visible(True)
    self.pauseVideo.set_sensitive(True)
    camrelease = False

    # t1 = threading.Thread(target=VideoPlayerWebcam, args=(self.outvideo1, self.outvideo2, self.taskcombobox, self.model1combobox, self.model2combobox))
    # t1.daemon = True
    # t1.start()



    t1 = threading.Thread(target=VideoPlayerWebcam, args=(self.outvideo1, self.taskcombobox, self.model1combobox, 1))
    t1.daemon = True
    t1.start()

    t2 = threading.Thread(target=VideoPlayerWebcam, args=(self.outvideo2, self.taskcombobox, self.model2combobox, 2))
    t2.daemon = True
    t2.start()

    # print("webcam radio toggled")

  def on_model2ComboBox_changed(self, widget, data=None):
    print("model2 combobox changed")
    # for c in self.data['tasks']:
    #   if c["task_name"] == self.selectedtask:
    #     for idx, c1 in enumerate(c["model"]):
    #       if c1["model_name"] == self.model1combobox.get_model()[self.model1combobox.get_active()][0]:
    #           model2file = c1["model_path"]
    #           model2config = c1["model_config_path"]

  def on_model1ComboBox_changed(self, widget, data=None):
    print("model1 combobox changed")
    model = widget.get_model()
    active = widget.get_active()
    if active >= 0:
      code = model[active][0]
      self.model2combobox.set_sensitive(True)
      self.model2store.clear()
      print(self.selectedtask)
      for c in self.data['tasks']:
          if c["task_name"] == self.selectedtask:
            for idx, c1 in enumerate(c["model"]):
              if c1["model_name"] != code:
                self.model2store.append([c1["model_name"], c1["model_name"]])
                
              if c1["model_name"] == code:
                  model1file = c1["model_path"]
                  model1config = c1["model_config_path"]

      self.model2combobox.set_active(0)
    

  def on_trainDataFileChooser_file_set(self, object, data=None):
    print("file chooser file set")

  def play(self, object, data=None):
    global play1, streamvideo
    play1 = True
    e = threading.Event()
    self.playVideo.set_sensitive(False)
    self.pauseVideo.set_sensitive(True)
    print(streamvideo)
    if streamvideo:
      if self.videoRadio.get_active():
        t1 = threading.Thread(name='videoplayer1' , target = VideoPlayer, args=(video_store + self.taskcombobox.get_model()[self.taskcombobox.get_active()][0] + "_" + self.model1combobox.get_model()[self.model1combobox.get_active()][0] + "_" + os.path.split(self.videochooserbutton.get_filename())[1].split('.')[0] + "_" + self.framerate.get_text() + ".avi", self.outvideo1, 1, e, self.playVideo, self.pauseVideo))
        t1.daemon = True
        t1.start()

        t2 = threading.Thread(name='videoplayer1', target = VideoPlayer, args=(video_store + self.taskcombobox.get_model()[self.taskcombobox.get_active()][0] + "_" + self.model2combobox.get_model()[self.model2combobox.get_active()][0] + "_" + os.path.split(self.videochooserbutton.get_filename())[1].split('.')[0] + "_" + self.framerate.get_text() + ".avi", self.outvideo2, 2, e, self.playVideo, self.pauseVideo))
        t2.daemon = True
        t2.start()
    else:
      play1 = True

  def pause(self, object, data=None):
    global play1, streamvideo
    streamvideo, play1= False, False
    # print(">>>>>>>>>>>>>>>>>>>>>" + str(t1.isAlive()))
    self.playVideo.set_sensitive(True)
    self.pauseVideo.set_sensitive(False)
    print("test")
  
  def on_taskComboBox_changed(self, widget, data=None):
    model = widget.get_model()
    active = widget.get_active()
    if active >= 0:
        code = model[active][0]
        self.selectedtask = code
        self.model1combobox.set_sensitive(True);
        self.model2combobox.set_sensitive(True);
        self.model1store.clear()
        self.model2store.clear()

        for c in self.data['tasks']:
          if c["task_name"] == code:
            for idx, c1 in enumerate(c["model"]):
              self.model1store.append([c1["model_name"], c1["model_name"]])
              if idx != 0:
                self.model2store.append([c1["model_name"], c1["model_name"]])
          
        self.model1combobox.set_active(0)
        # self.model2combobox.set_active(0)

        print('The selected task {}'.format(code))
    else:
        print('No task selected')



  def __init__(self):
    self.gladefile = "workbench-ui.glade"
    self.builder = gtk.Builder()
    self.builder.add_from_file(self.gladefile)
    self.builder.connect_signals(self)

    # Get objects
    go = self.builder.get_object
    self.taskstore = go('taskStore')
    self.model1store = go('model1Store')
    self.model2store = go('model2Store')
    self.taskcombobox = go('taskComboBox')
    self.model1combobox = go('model1ComboBox')
    self.model2combobox = go('model2ComboBox')
    self.videochooserbutton = go('videoChooserButton')
    self.outvideo1 = go('outVideo1')
    self.outvideo2 = go('outVideo2')
    self.model1proressbar = go('progressbar1')
    self.model2proressbar = go('progressbar2')
    self.generateresultmodel1 = go('generateResult1Button')
    self.generateresultmodel2 = go('generateResult2Button')
    self.playVideo = go('playVideo')
    self.pauseVideo = go('pauseVideo')
    self.framerate = go('framerate')
    self.processtime1 = go('processtime1')
    self.processtime2 = go('processtime2')
    self.model1processtime = go('model1processtime')
    self.model2processtime = go('model2processtime')
    self.webCamRadio = go('webCamRadio')
    self.videoRadio = go('videoRadio')

    # end_iter = self.processtime1.get_end_iter()
    # end_iter = self.processtime1.get_start_iter()
    # self.processtime1.insert(end_iter, "The text to insert at the end")
    # Initialize interface
    with open('./config.json') as f:
      self.data = json.load(f)

    for c in self.data['tasks']:
        self.taskstore.append([c["task_name"], c["task_name"]])
        self.selectedtask = c["task_name"]
    self.taskcombobox.set_active(0)

    self.window = self.builder.get_object("main_window")
    self.window.show()

if __name__ == "__main__":
  main = MLWB()
  gtk.main()
