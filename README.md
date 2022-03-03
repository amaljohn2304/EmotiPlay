# EmotiPlay
Plays music depending on the facial expression detected.<br/>

The Emotion detection can detect Anger, Surprise, Neutral, Sad and Hapiness<br/>
Songs have been made for Happy, Sad and Neutral<br/>
Setup
-----------
The program requires following packages to funtion :                                                                                                                               
pip3 install --user --upgrade tensorflow<br/><br/>
pip install opencv-python<br/>
pip install numpy<br/>
pip install playsound     //optional as i have used winsound which is inbuilt<br/>

Execution
----------------
Run the test.py file

Configuration
--------------------
In the Music folder each Folder contains music for the Respective emotion<br/>
More music can be added to the folder by following the pattern of naming observed in each folder<br/>
Adding additional folder will not have any effect as the program provides music for only 3 emotions<br/>
music files must be strictly .WAV files<br/>

It would be a good practice to compress WAV files using https://www.freeconvert.com/wav-compressor before adding it to the folders. <br/>

Feel free to modify to support more emotions and also feel free to add more music to these folders <br/>
