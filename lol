Basic Home Security using OpenCV and python
This article gives the basics on how to create a security system to detect intrusion. 
I’ll use webcam for the purpose of this article, but any camera can be used.
What is OpenCV?
OpenCV is a real-time computer vision library. In easy terms, OpenCV uses image(from a single frame of video/image itself) to understand the surroundings.
First we’ll turn on the webcam using opencv.
Python modules to install:
1. cv2( I have received some complaints that installing cv2 using pip is not supported for video)
2.imutils (pip install imutils)
Let’s open webcam using cv2.
https://gist.github.com/rohanricky/33a77f91925f7d10a1ac8b096e48586f
If you have encountered any problems with this code, it is mostly because OpenCV was not installed correctly.
A security system detects any movement in the camera environment. To achieve this, we will initialise a static frame and compare other frames with the static frame.
<script src=”https://gist.github.com/rohanricky/e81b9d663cb982cd64e268a027f4516b.js"></script>
Now we have our movement detection code running, can you sense some problems?
1. It doesn’t detect when at first I am in front of the camera and then move out. (told you “static frames”)
2. You told me it detects intruders, but it is detecting me also.( face recognition, we’ll get there)
Story time.
Initially when I made this code up, I was so excited that I setup my camera to watch out the front door. 
I didn’t catch/know the static frame thing back then, so while casually walking I opened the door and didn’t close it. Since in my first frame the door was closed, the code calculated the area and detected that there was an intruder. I also optimised the code to send me short videos to my Gmail. It so happens that the door got recorder in all the videos and Gmail blocked me for 5 GB data that it got repeatedly.
The lesson is static frame should be just that, static frame.
It was a disastrous problem. I couldn’t use it. Had to solve the issue.
What if instead of calculating the difference between first frame and the current frame we calculate the difference between previous frame and current frame? That way we can separate a static object and a moving object.
This is the modified code:
<script src=”https://gist.github.com/rohanricky/836b02653d9687f59ed921087f3fc2f7.js"></script>
This works a lot better than the previous one. Can also be used in real-world.
I’m working on a personal assistant(Jarvis), Home Automation and security projects. Please try them out and give me your valuable advice. Here are the links:
Jarvis: https://github.com/rohanricky/Jarvis
Home Automation : https://github.com/rohanricky/HomeAutomate
