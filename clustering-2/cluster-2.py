#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:32:14 2019

@author: ruben
"""

import face_recognition
import cv2
import pickle
import numpy as np
from datetime import datetime

LIMIT_FRAMES = 10
TOLERANCIA = 0.5
RATIO_FRAMES = 2

# Open the input movie file
input_movie = cv2.VideoCapture("./video.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps =  input_movie.get(cv2.CAP_PROP_FPS)

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# Load some sample pictures and learn how to recognize them.
known_faces = []
faces_nombre = []

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

identificador_face = 100

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break


    if ((frame_number%RATIO_FRAMES)==0):
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
    
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
        face_names = []
        ind_locations = 0
        for face_encoding in face_encodings:
            # REGLA NEGOCIO
            (x,y,x1,y1) = face_locations[ind_locations]
            ind_locations = ind_locations + 1
            area = abs(x1-x)*abs(y1-y)
            if (area < (90*90)):
                continue
            # See if the face is a match for the known face(s)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            if (len(face_distances)<=0):
                known_faces.append(face_encoding)
                faces_nombre.append(str(identificador_face))
                print (faces_nombre)
                identificador_face = identificador_face + 100
                continue
                
            best_match_index = np.argmin(face_distances)
            name = faces_nombre[best_match_index]
    
            if (min(face_distances)<=TOLERANCIA):
                known_faces.append(face_encoding)
                faces_nombre.append(name)
                face_names.append(name)
            else:
                known_faces.append(face_encoding)
                name = str(identificador_face)
                faces_nombre.append(name)
                identificador_face = identificador_face + 100
                face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    now = datetime.now() # current date and time
    print(now.strftime("%m/%d/%Y, %H:%M:%S"), "   Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
