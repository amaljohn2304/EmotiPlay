# EmotiPlay
Plays music depending on the facial expression detected.

The Emotion detection can detect Anger, Surprise, Neutral, Sad and Hapiness
Songs have been made for Happy, Sad and Neutral
##Setup
The program requires following packages to funtion :                                                                                                                               
--------pip3 install --user --upgrade tensorflow
--------
--------pip install opencv-python
--------
--------pip install numpy
---------
--------pip install playsound     //optional as i have used winsound which is inbuilt
--------
##Configuration
In the Music folder each Folder contains music for the Respective emotion
More music can be added to the folder by following the pattern of naming observed in each folder
Adding additional folder will not have any effect as the program provides music for only 3 emotions
music files must be strictly .WAV files

It would be a good practice to compress WAV files using https://www.freeconvert.com/wav-compressor before adding it to the folders. 

Feel free to modify to support more emotions and also feel free to add more music to these folders 
