import json
import os
import cv2

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

class Utils:

	def create_json(self, basepath, filename, data):
		try:
			fullpath = basepath + filename
			if not os.path.exists(fullpath):
				f = open(fullpath, "x")
				f.write(data)
				f.close
			else:
				f = open(fullpath, "w")
				f.write(data)
				f.close
		except Exception as e:
			print(e)

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

# if __name__ == "__main__":
# 	Utils().create_json("./analysis/", "test.json", "AA")