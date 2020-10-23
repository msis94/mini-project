import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time


def capture_data(haar_cascade_path, img_dataset, database_path):
    # take image from the camera
    cam = cv2.VideoCapture(0)

    # to know the FPS of the camera
    print(cam.get(cv2.CAP_PROP_FPS))

    # font to be used at rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX

    # minimum size to be recognized as a face
    minW = int(cam.get(3) / 3)
    minH = int(cam.get(4) / 3)

    face_detection = cv2.CascadeClassifier(haar_cascade_path)

    # Reading the database
    database = pd.read_csv(database_path)

    # Maximum image sample will be taken for each person
    max_sample = 30

    # For each person, enter one numeric face id (eg. 5)
    emp_num = input('\n Enter Employee Number : ')

    # Read all the data for column for employee number
    databases = str(database['Employee Number'].values)

    # Initialize individual sampling face count
    count = 0

    # if that employee number are not in databases then take the picture
    if emp_num not in databases:
        print("\n Please look at the camera and wait.....")
        while True:
            # Take image frame one by one
            ret, img = cam.read()

            # img = cv2.flip(img, -1) # flip video image vertically

            # change from color to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # detect face
            faces = face_detection.detectMultiScale(gray, 1.2, 5, minSize=(minW, minH))

            for (x, y, w, h) in faces:
                count += 1
                counter = count / 100 * 100

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.putText(img, str(counter)+"%", (x+5,y+h-5), font, 1, (255,255,0), 2)
                cv2.putText(img, str("{:.2f} %".format(counter)), (x + 5, y + h - 5), font, 1, (255, 255, 0), 2)

                # Save the captured image into the datasets folder
                cv2.imwrite(img_dataset + str(emp_num) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            # To make the window is sizeable
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)

            # To make window open full screen without title bar
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # show image with rectangle of detected faces
            cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                cam.release()
                cv2.destroyAllWindows()
                break
            elif count >= max_sample:  # Take 30 face sample and stop video
                cam.release()
                cv2.destroyAllWindows()
                break
    else:
        print("Employee number already exist please put another number")


def train_data(img_dataset, trained_model):
    # font to be used at rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize the haar cascade model
    face_detection = cv2.CascadeClassifier(haar_cascade_path)

    # Initialize the LPBH model
    face_recognition = cv2.face.LBPHFaceRecognizer_create()

    print("Please wait for model to train")

    # To join the link
    img_datasets = [os.path.join(img_dataset, f) for f in os.listdir(img_dataset)]

    # Empty array for face samples
    face_sample = []

    # Empty array for ids
    emp_num = []

    # initiate counter with zero
    count_sample = 0

    # Each image in the dataset in the image dataset will need to undergo LPBH training
    for image in img_datasets:

        # Read data in grayscale mode
        gray = cv2.imread(image, 0)

        # get the id only from the path name file
        id = int(os.path.split(image)[-1].split(".")[0])
        # print(id)
        faces = face_detection.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            # LBPH
            face_sample.append(gray[y:y + h, x:x + w])
            emp_num.append(id)

        count_sample += 1
        progress = count_sample / len(img_datasets) * 100
        print("progress to train {:.2f} %".format(progress))
        # -------------------------------------
    face_recognition.train(face_sample, np.array(emp_num))

    # Save the model into trainer/trainer.yml
    face_recognition.save(trained_model)

    # Print the number of faces has been trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(emp_num))))


def recognize(haar_cascade_path, trained_model, database_path):
    home = str(Path.home()) + "/Desktop/facial_recognition"

    cam = cv2.VideoCapture(home + '/ola.webm')
    print(cam.get(cv2.CAP_PROP_FPS))

    # font to be used at rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX

    # MINIMUM SIZE TO BE RECOGNIZED AS A FACE
    minW = int(cam.get(3) / 3)
    minH = int(cam.get(4) / 3)
    face_detection = cv2.CascadeClassifier(haar_cascade_path)

    # 3. Path for LBPH pretrained model
    face_recognition = cv2.face.LBPHFaceRecognizer_create()

    # 4. Path of employee database
    database = pd.read_csv(database_path)
    min_acc = 60
    # READ WEIGHT OF LPBH TRAINED MODEL
    face_recognition.read(trained_model)

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0
    fpsanalysis = []

    while True:
        start_time = time.time()  # start time of the loop
        ret, img = cam.read()

        if img is None:
            print("Minimum FPS: {:.0f}".format(min(fpsanalysis)))
            print("Maximum FPS: {:.0f}".format(max(fpsanalysis)))
            print("Average FPS: {:.0f}".format(sum(fpsanalysis)/len(fpsanalysis)))
            print(len(fpsanalysis))
            cv2.destroyAllWindows()

        #     img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detection.detectMultiScale(gray, 1.2, 5, minSize=(minW, minH))

        # this is timer for cases where it does not detect the faces
        result = 1.0 / (time.time() - start_time)
        strs = "FPS : {:.0f}".format(result)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # LPBH
            emp_num, confidence = face_recognition.predict(gray[y:y + h, x:x + w])

            # --------------------------------------------
            # find the row of employee number (eg. = 7)
            emp_name = database.loc[database["Employee Number"] == emp_num]

            # after found employee number show the name of the employee by taking
            # the -name- column
            emp_name = emp_name["Employee Name"].values[0]

            acc = round(100 - confidence)

            if acc > min_acc:
                emp_name
            else:
                emp_name = "unknown"

            # cv2.putText(img, 'Izham Pass : Door Opens', (10,40), font, 1, (255,255,255), 2)
            cv2.putText(img, str(emp_name), (x, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, "{} %".format(str(acc)), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            result = 1.0 / (time.time() - start_time)
            strs = "FPS : {:.0f}".format(result)  # FPS = 1 / time to process loop
            # fpsanalysis.append(result)
        # if acc > min_acc:
        #     fpsanalysis.append(result)
        # time when we finish processing for this frame
        new_frame_time = time.time() //10.80
        print(new_frame_time)
        fps = 1/(new_frame_time-prev_frame_time)
        # print(fps)
        prev_frame_time = new_frame_time //10.40

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        # puting the FPS count on the frame
        cv2.putText(img, fps, (7, 200), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.putText(img, str(strs), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        # ----------------------------------------

        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
        cv2.imshow('image', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            # print(min(fpsanalysis))
            # print(max(fpsanalysis))
            # print(stop - start)
            cam.release()
            cv2.destroyAllWindows()
            break


while True:

    # To know machine path and file path
    home = str(Path.home()) + "/Desktop/facial_recognition"

    # The location of haar cascade file
    haar_cascade_path = home + '/haarcascade_frontalface_default.xml'

    # The location of trained model file
    trained_model = home + '/trainer.yml'

    # The location of image dataset
    img_dataset = home + '/dataset/'

    # The location of face database file which is contain name and numbers
    database_path = home + "/face_database.csv"

    # Show popup to choose whether to capture/train/deploy new data or to recognize
    try:
        choice = int(input("\n1. Capture data"
                           "\n2. Train data"
                           "\n3. Recognition face"
                           "\n\nEnter your choice : "))

        if choice == 1:
            # the face of the person will be captured by using haar cascade technique and
            # the cropped face will save in the image dataset folder, each image will be given
            # label and the label of the image will save in the database
            capture_data(haar_cascade_path, img_dataset, database_path)

        elif choice == 2:
            # train the data which is face recognition using LBPH technique and save the model
            # in the folder trained model
            train_data(img_dataset, trained_model)

        elif choice == 3:
            # This phase is used to deploy the system, that means the face already trained
            # In this case we will use the haar cascade to detect the face and LBPH to recognize
            # that face and find the label of that face inside the database folder
            recognize(haar_cascade_path, trained_model, database_path)

        else:
            exit()

    except Exception as e:
        print(e)
