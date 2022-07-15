# Smile Recognition
Using machine learning and facial recognition to predict if an image from a live webcam is smiling or not. Uses Python, OpenCV, and Scikit learn.

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Caleb | Saratoga High School | Machine Learning | Incoming Sophomore |


![Headstone Image](https://lh3.googleusercontent.com/pw/AM-JKLWubDX9DI7mwGB0GznaocIjVxcxtainTurdJ3fOPy5NRXLLj4pDch6dC1Qk9dxsGlz61ufM4PvHH9JDj3iaYgpjzVn-Yt5OD8r2lnaCjcHA15QkBZKB-dYSp41W2ei77V1s7yR5B0-JXfX0EDRXHGZb=w1644-h1642-no?authuser=0)
  
# Milestone 3: Accuracy and Frame rate
My final milestone is the increased reliability and accuracy of my robot. After milestone #2 I searched through my code trying to find out why it was running so slow. I realize that it was because I had a 2 for loops in a while loop. This basically means that my code runs exponentially slower. I fixed this by creating a seperate thread to run the functions of extracting, detecting, and predicting that runs every 15 frames. This allows the video feed to come in faster, so I get around 10-15 fps now. Besides the frame rate, I also ran into a problem with the accuracy of the program. This was because the detected and extracted face that the program got was not cropped and resized correctly. So I created a different file to test the code using images. I learned how to crop 2d arrays by slicing and was able to make the machine more accurate.

[![Final Milestone](https://img.youtube.com/vi/IykI0vbSTy8/maxresdefault.jpg )](https://www.youtube.com/watch?v=IykI0vbSTy8 "Final Milestone"){:target="_blank" rel="noopener"}

# Milestone 2: Face detection, extraction, and prediction
My second milestone was face detection, extraction and prediction through OpenCV. The documentation given to me for face detection was outdated and buggy, so I decided to search the web for a better option. After finding some reasources I created an empty file and wrote my own facial detection software compatible to my current project. I then tested it and after it worked I implemented it into my code. For the extraction of the face, there were bits of the documentation that were completely unessecary and just created problems, so I deleted them. I also had a problem with matching the number of features from the extracted image and how many svc needs. To solve this I mapped the image onto a 4096 by 4096 2d array. After this, I created a loop that extracts frames from the webcam and runs it through extraction and prediction functions. The code runs, but as of now I get a total of 0.2 fps. 


[![Third Milestone](https://i3.ytimg.com/vi/zEV1-paEMxw/maxresdefault.jpg)](https://www.youtube.com/watch?v=zEV1-paEMxw){:target="_blank" rel="noopener"}

# Milestone 1: Fetching data and traning the machine
My first milestone was loading in the training dataset and training the machine. Originally, I was supposed to create an UI to classify all 400 faces my self, but I found that too tedious. I decided to load in an already trained dataset and train the machine from there. I used a series of functions in the sklearn libraries to train my computer. Everything went pretty smoothly and now I am able to predict if a photo is smiling or not. 

[![First Milestone](https://i3.ytimg.com/vi/Co-QWS_dc-s/maxresdefault.jpg)](https://www.youtube.com/watch?v=Co-QWS_dc-s){:target="_blank" rel="noopener"}



# Starter Project (Big Time Watch)
The starter project that I made was the the Big Time Watch. This wasn't my original project, but due to me screwing up the first one, I ended up with the watch. . I ran into a few problems while making this project. The first was the broken pins on the microcontroller. I substituted them with stripped solid core wires, which I had to solder at an angle to connect them to the microcontroller. The second problem I face was connectivity issues. Some of my pins weren't soldered in well, so I had to go and fix that. I had a lot fun making this project, even with the various problems I encountered, and really enjoyed seeing it work in the end. 

[![Starter Project](https://i3.ytimg.com/vi/whtRA3-QsV8/maxresdefault.jpg)](https://www.youtube.com/watch?v=whtRA3-QsV8){:target="_blank" rel="noopener"}
