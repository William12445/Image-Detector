from Dector import *
import os

def main():
    videoPath = 0  # If you want to use a webcam, otherwise provide the path to the video file

    base_dir = 'd:/230073/real_time_object_detection_cpu-main-20240607T015456Z-001/real_time_object_detection_cpu-main'

    configPath = os.path.join(base_dir, "model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join(base_dir, "model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join(base_dir, "model_data", "coco.names")
    
    detector = Dector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == "__main__":
    main()
