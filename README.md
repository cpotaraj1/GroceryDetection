# gordianRobotics
Tried U2Seg and SAM model to get an estimate of ROI from th eimage but it needed a lot more time to tune it. With the time constraints in mind, I have decided to move with yolov5 and train the model to do TASK 1

This repo helps in understanding how we used yolov5 weights trained on imagenet data and tuned the model on SKU110 dataset
We have the following results of the model against the unseen data:
![Master test image given in the task( detected all instances of the objects)](images/master_test.JPG)
![Alt text](images/Results/val_0.jpg) ![Alt text](images/Results/val_1.jpg) ![Alt text](images/Results/val_3.jpg) ![Alt text](images/Results/val_15.jpg) ![Alt text](images/Results/val_24.jpg) ![Alt text](images/Results/val_58.jpg) ![Alt text](images/Results/val_59.jpg) ![Alt text](images/Results/val_66.jpg) ![Alt text](images/Results/val_112.jpg) ![Alt text](images/Results/val_128.jpg)


SPEED TEST:
The complete breakdown of the inference time 
Speed: 0.1ms pre-process, 4.3ms inference, 2.2ms NMS per image at shape (16, 3, 640, 640)

Some useful graphs from training session
![Alt text](images/validation_metrics.png) ![Alt text](images/validation_loss.png) ![Alt text](images/training_loss.png)
![confusion matrix](images/training/confusion_matrix.png) ![F1 Curve](images/training/F1_curve.png) ![Precision Curve](images/training/P_curve.png) ![Precision Recall curve](images/training/PR_curve.png) ![Recall curve](images/training/R_curve.png)

Install requirements.txt and run "python detect.py"

TO-DO TASKS:
Add training scripts 
add validation scripts
add augmentation scripts
Need to test aws, docker and flask api utilities in "util" folder
Need to make sure the pytorch models are hosted in google drive and a downloadable link is avaialble
