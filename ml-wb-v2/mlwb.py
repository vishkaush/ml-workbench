import sys
import cv2
from PyQt5.QtCore import QTimer,QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication,QDialog,QMainWindow,QWidget, QLabel,QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont
from PyQt5 import QtCore, QtGui, QtWidgets
import dlib
import os
import cv2
import numpy as np
from imutils.video import VideoStream
import imutils


import json
video_source=0
analysisrate=1
isPaused=0
Task=""
#Model 1 Parameters:
model1_Name=""
model1_configPath=""
model1_filePath=""



#Model 2 Parameters:
model2_Name=""
model2_configPath=""
model2_filePath=""
class MLWB(QMainWindow):
    
    
    def __init__(self):
        super(MLWB,self).__init__()
        loadUi('mlwbui.ui',self)
        self.image=None
        self.processedImage=None
        self.PlayButton.clicked.connect(self.PlayVideo)
        self.PauseButton.clicked.connect(self.PauseVideo)
        #self.detectButton.setCheckable(True)
        #self.detectButton.toggled.connect(self.detect_webcam_face)
        self.face_Enabled=False
        self.detected_objects = []
        self.FileChooserButton.setEnabled(False)
        self.FileChooserButton.clicked.connect(self.OpenFileDialog)

        self.model = QStandardItemModel()
        self.Tasks_ComboBox.setModel(self.model)
        self.Model1_ComboBox.setModel(self.model)
        self.Model2_ComboBox.setModel(self.model)
        self.model1_name=""
        self.model2_name=""
        self.model1_configpath=""
        self.model2_configpath=""
        self.model1_filepath=""
        self.model2_configpath=""
        self.filepath=""
        self.source=0
        self.WebCam_RadioButton.toggled.connect(self.Radiobutton_Toggled)
        self.VideoFile_RadioButton.toggled.connect(self.Radiobutton_Toggled)
        
        #self.faceCascade=''
        
        self.task_selected=""
        self.video_writer1=""
        self.video_filename1=""
        self.video_writer2=""
        self.video_filename2=""
        self.genResultModel1Button.clicked.connect(self.generate_Result_Model_1_clicked)
        self.genResultModel2Button.clicked.connect(self.generate_Result_Model_2_clicked)
        #self.genResultModel2Button.setEnabled(True)
        
        global analysisrate

        
        

        


        
        with open('./config.json') as f:
            self.data=json.load(f)
            for c in self.data['tasks'] :
                task=QStandardItem(c['task_name'])
                self.model.appendRow(task)
                for model in c['model']:
                    model_name=QStandardItem(model['model_name'])
                    task.appendRow(model_name)
                
                
        self.Tasks_ComboBox.currentIndexChanged.connect(self.updateStateCombo)
        self.updateStateCombo(0)
        
        
    def OpenFileDialog(self):
        
        filename = QFileDialog.getOpenFileName()
        
        self.source = filename[0]
        print(self.source)

       

        
        
    def Radiobutton_Toggled(self):
        if self.WebCam_RadioButton.isChecked():
            print("Wc")
            self.source=0
            
        if self.VideoFile_RadioButton.isChecked():
            self.FileChooserButton.setEnabled(True)
    
    
        

    def setImage1(self, image):
        
        self.VideoPlayer_Model1.setPixmap(QPixmap.fromImage(image))   
        

    def setImage2(self, image):
        
        self.VideoPlayer_Model2.setPixmap(QPixmap.fromImage(image))   
        
    
        
    def updateStateCombo(self, index):
        
        indx = self.model.index(index, 0, self.Tasks_ComboBox.rootModelIndex())
        self.Model1_ComboBox.setRootModelIndex(indx)
        self.Model1_ComboBox.setCurrentIndex(0) 
        self.Model2_ComboBox.setRootModelIndex(indx)
        self.Model2_ComboBox.setCurrentIndex(1) 
        self.Model1_ComboBox.currentIndexChanged.connect(self.model_selector)
        self.Model2_ComboBox.currentIndexChanged.connect(self.model_selector)
        
        
        


    ''' def detect_webcam_face(self,status):
            if status:
                self.detectButton.setText('Stop Detection')
                self.face_Enabled=True
            else:
                self.detectButton.setText('Detect Face') 
                self.face_Enabled=False
'''
    def PlayVideo(self):
        pass
    
    
        

    def generate_Result_Model_1_clicked(self):
        #self.capture=cv2.VideoCapture(self.source)
        #self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        #self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        global analysisrate
        analysisrate=self.AnalysisRate_TextBox.toPlainText()
        print(analysisrate)
        global video_source
        self.th1 = Thread_Model1(self)
        self.progressBarModel1.setRange(0,100)
        self.th1.valueChanged.connect(self.progressBarModel1.setValue)
        
        #isPaused=0
        #self.PauseButton.setEnabled(True)
        #srcc1="/Users/aniketpihu/Downloads/bionic.mp4"
        video_source=self.source
        print(video_source)
        self.th1.changePixmap.connect(self.setImage1)
        
        
        self.th1.start()
     


        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
            
            
            
            

    def generate_Result_Model_2_clicked(self):
        #self.capture=cv2.VideoCapture(self.source)
        #self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        #self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        global analysisrate
        analysisrate=self.AnalysisRate_TextBox.toPlainText()
        print(analysisrate)
        global video_source
        self.th2 = Thread_Model2(self)
        
        self.th2.valueChanged.connect(self.progressBarModel2.setValue)
        
        #self.PauseButton.setEnabled(True)
        #srcc1="/Users/aniketpihu/Downloads/bionic.mp4"
        video_source=self.source
        print(video_source)
        self.th2.changePixmap.connect(self.setImage2)
        self.th2.start()
        


        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()

    def update_frame(self):
        ret,self.image=self.capture.read()
        
        self.image=cv2.flip(self.image,1)
        
        
        
        
        detected_image_1=self.detect_face_model1(self.image)
        self.video_writer1=cv2.VideoWriter('output-{}-{}.avi'.format(self.model1_name,str(self.source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (detected_image_1.shape[1],detected_image_1.shape[0]))
        self.video_writer1.write(detected_image_1)
        self.displayImage(detected_image_1,1)
        detected_image_2=self.detect_face_model2(self.image)
        self.video_writer2=cv2.VideoWriter('output-{}-{}.avi'.format(self.model2_name,str(self.source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (detected_image_1.shape[1],detected_image_1.shape[0]))
        self.video_writer2.write(detected_image_2)
        self.displayImage(detected_image_2,2)
            
    #To select Models from ComboBox
    def model_selector(self,i):
        global Task,model1_Name,model2_Name,model1_filePath,model1_configPath,model2_filePath,model2_configPath
        Task=self.Tasks_ComboBox.currentText()
        model1=self.Model1_ComboBox.currentText()
        model2=self.Model2_ComboBox.currentText()
        for tasks in self.data['tasks']:
            if tasks['task_name'] == Task :
                #Updating Model 1 parameters
                for item in tasks['model']:
                    if item['model_name']==model1:
                        model1_Name=model1
                        model1_filePath=os.getcwd()+item['model_path']
                        model1_configPath=os.getcwd()+item['model_config_path']
                        print(model1_Name)
                        print(model1_configPath)
                
                        
                
                #Updating Model 2 parameters
                for item in tasks['model']:
                    if item['model_name']==model2:
                        model2_Name=model2
                        model2_filePath=os.getcwd()+item['model_path']
                        model2_configPath=os.getcwd()+item['model_config_path']
                        print(model2_Name)
                        print(model2_configPath)
                        
                        #print(self.model2_configfile)
                        
                    
    
    
    
    
    def PauseVideo(self):
        global video_writer1,video_writer2
        cv2.destroyAllWindows()
        video_writer1.release()
        video_writer2.release()
        
    


    def displayImage(self,img,window=1):
       pass
            
            
            
           
            
            
            
            

class Thread_Model1(QThread):
    changePixmap = pyqtSignal(QImage)
    global video_source
    f=0;
    valueChanged = QtCore.pyqtSignal(int)
    processFinished=pyqtSignal(int)


    
    def run(self):
        
        
        framecount = 0
        
    
        global isPaused,analysisrate,Task,model1_Name,model1_configPath,model1_filePath,model2_Name,model2_configPath,model2_filePath,video_writer1
        self.model1_name=model1_Name
        self.model2_name=model2_Name
        
        
        self.capture = cv2.VideoCapture(video_source)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {}".format(self.capture.get(cv2.CAP_PROP_FPS)))
        print(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        #starttime = time.time()
        ret, self.frame = self.capture.read()
        video_writer1=cv2.VideoWriter('output-{}-{}.avi'.format(model1_Name,str(video_source).split('/')[-1]),cv2.VideoWriter_fourcc('M','J','P','G'),int(analysisrate), (self.frame.shape[1],self.frame.shape[0]))
        
        while self.capture.isOpened():
            
            ret, self.frame = self.capture.read(framecount)
            if framecount <= int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                framecount += int(analysisrate)
                rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                self.ret_img1=rgbImage
                    
                    
                    
                    
                if(Task=="Face Detection"):
                    
                    self.ret_img1=self.detect_face_model1(rgbImage)
                        
                    
                    
                    
                    
                bgrImage=   cv2.cvtColor(self.ret_img1, cv2.COLOR_RGB2BGR)
                video_writer1.write(bgrImage)
                convertToQtFormat = QImage(self.ret_img1.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480)#Qt.KeepAspectRatio
                    
                
                self.changePixmap.emit(p)
                nextFrameNo = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                totalFrames =self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                complete = 100*(nextFrameNo/totalFrames)*(float(analysisrate))
                
                print(complete)
                self.valueChanged.emit(complete)
                    
                    
                
                    
                    
                k = cv2.waitKey(0)
                if k == 27:  
                    cv2.destroyAllWindows()
                        
                    
                    
                    
    def detect_face_model1(self,img):
        
        ret_img=img
        
        global isPaused,analysisrate,Task,model1_Name,model1_configPath,model1_filePath,model2_Name,model2_configPath,model2_filePath
        print(model1_Name)
        if model1_Name=="Haar Cascade":
            
            
            
            
            ret_img=self.Haar_Cascade(img,file_path=model1_filePath)
            
        if model1_Name=="MMOD Dlib" or model1_Name=="HOG Dlib":
            ret_img=self.Dlib(img,file_path=model1_filePath)
            
            
            
        if model1_Name=="MTCNN":
            ret_img=self.MTCNN(img,configFile=model1_configPath,modelFile=model1_filePath)
            
            
            
        return ret_img
  
    

                
                
    def Haar_Cascade(self,img,file_path):
        
        #modelfilee='/Users/aniketpihu/ml-workbench-v2/Models/Face_Detection/models/haarcascade_frontalface_default.xml'
        
        faceCascade=cv2.CascadeClassifier(file_path)
         
        after_image=img.copy()
        gray=cv2.cvtColor((after_image),cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray,1.2,5,minSize=(90,90))

        for(x,y,w,h) in faces:
            cv2.rectangle(after_image,(x,y),(x+w,y+h),(255,0,0),2)

        return after_image
    
    
    def Dlib(self, img, inHeight=300, inWidth=0,file_path=""):
        mymodel=""
        
        if self.model1_name=="MMOD Dlib" or self.model2_name=="MMOD Dlib":
            detector=dlib.cnn_face_detection_model_v1(file_path)
     
            
        if self.model1_name=="HOG Dlib" or self.model2_name=="HOG Dlib":
            mymodel="HOG Dlib"
            detector=dlib.get_frontal_face_detector()

        frameDlib = img.copy()
        frameHeight = frameDlib.shape[0]
        frameWidth = frameDlib.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight)*inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameDlibSmall = cv2.resize(frameDlib, (inWidth, inHeight))

        frameDlibSmall = cv2.cvtColor(frameDlibSmall, cv2.COLOR_BGR2RGB)
        
        faceRects = detector(frameDlibSmall, 0)

        #print(frameWidth, frameHeight, inWidth, inHeight)
        bboxes = []
        if mymodel=="HOG Dlib":
            for faceRect in faceRects:
                cvRect = [int(faceRect.left()*scaleWidth), int(faceRect.top()*scaleHeight),
                      int(faceRect.right()*scaleWidth), int(faceRect.bottom()*scaleHeight) ]
                bboxes.append(cvRect)
                print(bboxes)
                cv2.rectangle(frameDlib, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (255, 0, 0),int(round(frameHeight/150)), 2)
                
                
                
        else:
                
                for faceRect in faceRects:
                    cvRect = [int(faceRect.rect.left()*scaleWidth), int(faceRect.rect.top()*scaleHeight),
                              int(faceRect.rect.right()*scaleWidth), int(faceRect.rect.bottom()*scaleHeight) ]
                    bboxes.append(cvRect)
                    print(bboxes)
                    cv2.rectangle(frameDlib, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (255, 0, 0),int(round(frameHeight/150)), 2)
        
        
        
        
        return frameDlib
    
    
     
    
    
    
    
    def DNN(self,frame,configFile,modelFile ):
        #if (self.task=="Face Detection" and (self.model1_name=="MTCNN" or self.model2_name=="MTCNN")):
        net=cv2.dnn.readNetFromCaffe(configFile, modelFile)
        
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.9:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
                
                
                
        return frameOpencvDnn
        
    
    
    
    

class Thread_Model2(QThread):
    changePixmap = pyqtSignal(QImage)
    global video_source
    f=0;
    valueChanged = QtCore.pyqtSignal(int)


    
    def run(self):
        
        
        framecount = 0
    
        global isPaused,analysisrate,Task,model2_Name,model2_configPath,model2_filePath,video_writer2
        
        self.model2_name=model2_Name
        
       
        
        self.capture = cv2.VideoCapture(video_source)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {}".format(self.capture.get(cv2.CAP_PROP_FPS)))
        print(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        #starttime = time.time()
        ret, self.frame = self.capture.read()
        video_writer2=cv2.VideoWriter('output-{}-{}.avi'.format(model2_Name,str(video_source).split('/')[-1]),cv2.VideoWriter_fourcc('M','J','P','G'),int(analysisrate), (self.frame.shape[1],self.frame.shape[0]))
        
        while self.capture.isOpened():
            
            ret, self.frame = self.capture.read(framecount)
            if framecount <= int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                framecount += int(analysisrate)
                rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                self.ret_img2=rgbImage
                    
                    
                    
                    
                if(Task=="Face Detection"):
                    
                    self.ret_img2=self.detect_face_model2(rgbImage)
                        
                    
                    
                
                
                bgrImage=   cv2.cvtColor(self.ret_img2, cv2.COLOR_RGB2BGR)
                video_writer2.write(bgrImage)
                convertToQtFormat = QImage(self.ret_img2.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480)#Qt.KeepAspectRatio
                    
                
                self.changePixmap.emit(p)
                nextFrameNo = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                print("NFn")
                print(nextFrameNo)
                totalFrames =self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                print(totalFrames)
                complete = 100*((nextFrameNo/totalFrames))*((float(analysisrate)))
                
                self.valueChanged.emit(complete)
                    
                k = cv2.waitKey(0)
                if k == 27:  
                    cv2.destroyAllWindows()
                        
                    
                    

            
            
     
    def detect_face_model2(self,img):
        global isPaused,analysisrate,Task,model1_Name,model1_configPath,model1_filePath,model2_Name,model2_configPath,model2_filePath,video_writer1
        ret_img=img
        
        if model2_Name=="Haar Cascade":
            
            
            
            ret_img=self.Haar_Cascade(img,model2_filePath)
            
        if model2_Name=="MMOD Dlib" or model2_Name=="HOG Dlib":
            ret_img=self.Dlib(img,file_path=model2_filePath)
            
        if model2_Name=="MTCNN":
            ret_img=self.MTCNN(img,configFile=model2_configPath,modelFile=model2_filePath)
            
            
        return ret_img
    

                
                
    def Haar_Cascade(self,img,file_path):
        
        
        #modelfilee='/Users/aniketpihu/ml-workbench-v2/Models/Face_Detection/models/haarcascade_frontalface_default.xml'
        
        faceCascade=cv2.CascadeClassifier(file_path)
         
        after_image=img.copy()
        gray=cv2.cvtColor((after_image),cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray,1.2,5,minSize=(90,90))

        for(x,y,w,h) in faces:
            cv2.rectangle(after_image,(x,y),(x+w,y+h),(255,0,0),2)

        return after_image
    
    
    def Dlib(self, img, inHeight=300, inWidth=0,file_path=""):
        mymodel=""
        
        if self.model2_name=="MMOD Dlib":
            detector=dlib.cnn_face_detection_model_v1(file_path)
     
            
        if self.model2_name=="HOG Dlib":
            mymodel="HOG Dlib"
            detector=dlib.get_frontal_face_detector()

        frameDlib = img.copy()
        frameHeight = frameDlib.shape[0]
        frameWidth = frameDlib.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight)*inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameDlibSmall = cv2.resize(frameDlib, (inWidth, inHeight))

        frameDlibSmall = cv2.cvtColor(frameDlibSmall, cv2.COLOR_BGR2RGB)
        
        faceRects = detector(frameDlibSmall, 0)

        #print(frameWidth, frameHeight, inWidth, inHeight)
        bboxes = []
        if mymodel=="HOG Dlib":
            for faceRect in faceRects:
                cvRect = [int(faceRect.left()*scaleWidth), int(faceRect.top()*scaleHeight),
                      int(faceRect.right()*scaleWidth), int(faceRect.bottom()*scaleHeight) ]
                bboxes.append(cvRect)
                print(bboxes)
                cv2.rectangle(frameDlib, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (255, 0, 0),int(round(frameHeight/150)), 2)
                
                
                
        else:
                
                for faceRect in faceRects:
                    cvRect = [int(faceRect.rect.left()*scaleWidth), int(faceRect.rect.top()*scaleHeight),
                              int(faceRect.rect.right()*scaleWidth), int(faceRect.rect.bottom()*scaleHeight) ]
                    bboxes.append(cvRect)
                    print(bboxes)
                    cv2.rectangle(frameDlib, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (255, 0, 0),int(round(frameHeight/150)), 2)
        
        
        
        
        return frameDlib
    
    
     
    
    
    
    
    def DNN(self,frame,configFile,modelFile ):
        #if (self.task=="Face Detection" and (self.model1_name=="MTCNN" or self.model2_name=="MTCNN")):
        net=cv2.dnn.readNetFromCaffe(configFile, modelFile)
        
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.9:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
                
                
                
        return frameOpencvDnn
        
    
    

    
    


if __name__=='__main__':
    app=QApplication(sys.argv)
    window=MLWB()
    window.setWindowTitle('ML WB')
    window.show()
    sys.exit(app.exec_())
    global video_writer1
    
