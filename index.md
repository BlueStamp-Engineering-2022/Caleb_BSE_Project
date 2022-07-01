# Smile Recognition
Using machine learning and facial recognition to predict if an image from a live webcam is smiling or not. Uses Python, OpenCV, and Scikit learn.

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Caleb Yu | Saratoga High School | Machine Learning | Incoming Sophomore |


![Headstone Image](https://lh3.googleusercontent.com/pw/AM-JKLWubDX9DI7mwGB0GznaocIjVxcxtainTurdJ3fOPy5NRXLLj4pDch6dC1Qk9dxsGlz61ufM4PvHH9JDj3iaYgpjzVn-Yt5OD8r2lnaCjcHA15QkBZKB-dYSp41W2ei77V1s7yR5B0-JXfX0EDRXHGZb=w1644-h1642-no?authuser=0)
  
# Milestone 3: Real time facial detection
My final milestone is the increased reliability and accuracy of my robot. I ameliorated the sagging and fixed the reliability of the finger. As discussed in my second milestone, the arm sags because of weight. I put in a block of wood at the base to hold up the upper arm; this has reverberating positive effects throughout the arm. I also realized that the forearm was getting disconnected from the elbow servo’s horn because of the weight stress on the joint. Now, I make sure to constantly tighten the screws at that joint. 

[![Final Milestone](https://res.cloudinary.com/marcomontalbano/image/upload/v1612573869/video_to_markdown/images/youtube--F7M7imOVGug-c05b58ac6eb4c4700831b2b3070cd403.jpg )](https://www.youtube.com/watch?v=F7M7imOVGug&feature=emb_logo "Final Milestone"){:target="_blank" rel="noopener"}

# Milestone 2: Face detection, extraction, and prediction
My second milestone was face detection, extraction and prediction through OpenCV. The documentation given to me for face detection was outdated and buggy, so I decided to search the web for a better option. After finding some reasources I created an empty file and wrote my own facial detection software compatible to my current project. I then tested it and after it worked I implemented it into my code. For the extraction of the face, there were bits of the documentation that were completely unessecary and just created problems, so I deleted them. I also had a problem with matching the number of features from the extracted image and how many svc needs. To solve this I mapped the image onto a 4096 by 4096 2d array. After this, I created a loop that extracts frames from the webcam and runs it through extraction and prediction functions. The code runs, but as of now I get a total of 0.2 fps. 


[![Third Milestone](https://i3.ytimg.com/vi/zEV1-paEMxw/maxresdefault.jpg)](https://www.youtube.com/watch?v=zEV1-paEMxw){:target="_blank" rel="noopener"}

# Milestone 1: Fetching data and traning the machine
My first milestone was loading in the training dataset and training the machine. Originally, I was supposed to create an UI to classify all 400 faces my self, but I found that too tedious. I decided to load in an already trained dataset and train the machine from there. I used a series of functions in the sklearn libraries to train my computer. Everything went pretty smoothly and now I am able to predict if a photo is smiling or not. 

[![First Milestone](https://i3.ytimg.com/vi/Co-QWS_dc-s/maxresdefault.jpg)](https://www.youtube.com/watch?v=Co-QWS_dc-s){:target="_blank" rel="noopener"}



# Starter Project (Big Time Watch)
The starter project that I made was the the Big Time Watch. This wasn't my original project, but due to me screwing up the first one, I ended up with the watch. . I ran into a few problems while making this project. The first was the broken pins on the microcontroller. I substituted them with stripped solid core wires, which I had to solder at an angle to connect them to the microcontroller. The second problem I face was connectivity issues. Some of my pins weren't soldered in well, so I had to go and fix that. I had a lot fun making this project, even with the various problems I encountered, and really enjoyed seeing it work in the end. 

[![Starter Project](https://i3.ytimg.com/vi/whtRA3-QsV8/maxresdefault.jpg)](https://www.youtube.com/watch?v=whtRA3-QsV8){:target="_blank" rel="noopener"}
