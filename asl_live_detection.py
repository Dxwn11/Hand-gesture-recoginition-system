import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("Starting ASL Gesture Recognition System")

model = load_model("asl_model.h5")

labels = [
"A","B","C","D","E","F","G","H","I","J","K","L","M",
"N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
"del","nothing","space"
]

reference = cv2.imread("Gesture/american_sign_language.PNG")
reference = cv2.resize(reference,(420,320))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

current_word = ""
letter = ""
confidence = 0


# Mouse click for clear button
def mouse_click(event,x,y,flags,param):
    global current_word

    if event == cv2.EVENT_LBUTTONDOWN:
        if 50 < x < 200 and 250 < y < 300:
            current_word = ""
            print("Word cleared")


cv2.namedWindow("ASL AI Dashboard")
cv2.setMouseCallback("ASL AI Dashboard", mouse_click)


while True:

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame,1)

    h, w, _ = frame.shape

    # ROI box
    x1, y1 = 100,100
    x2, y2 = 350,350
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    roi = frame[y1:y2, x1:x2]

    roi_resized = cv2.resize(roi,(64,64))
    roi_resized = roi_resized / 255.0
    roi_resized = roi_resized.reshape(1,64,64,3)

    prediction = model.predict(roi_resized, verbose=0)

    index = np.argmax(prediction)
    confidence = prediction[0][index]*100

    if index < len(labels):
        letter = labels[index]
    else:
        letter = "?"


    # -------- CAMERA + CHART --------

    camera_ui = np.zeros((h, w+430,3), dtype=np.uint8)
    camera_ui[0:h,0:w] = frame

    camera_ui[40:360,w+5:w+425] = reference

    cv2.putText(camera_ui,
                "ASL Reference Chart",
                (w+90,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2)


    # -------- DASHBOARD --------

    dashboard = np.zeros((350,500,3), dtype=np.uint8)

    cv2.putText(dashboard,
                "Live Translation",
                (140,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2)

    cv2.putText(dashboard,
                f"Word: {current_word}",
                (50,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.putText(dashboard,
                f"Prediction: {letter}",
                (50,160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255,255,255),
                2)

    cv2.putText(dashboard,
                f"Confidence: {confidence:.2f}%",
                (50,210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,0),
                2)

    # confidence bar
    bar_width = int(confidence*3)

    cv2.rectangle(dashboard,(50,230),(350,250),(60,60,60),-1)
    cv2.rectangle(dashboard,(50,230),(50+bar_width,250),(0,255,0),-1)


    # CLEAR button
    cv2.rectangle(dashboard,(50,250),(200,300),(0,0,255),-1)

    cv2.putText(dashboard,
                "CLEAR WORD",
                (65,285),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2)


    cv2.putText(dashboard,
                "Press C to capture letter",
                (260,280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200,200,200),
                1)

    cv2.putText(dashboard,
                "Press ESC to exit",
                (260,310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200,200,200),
                1)


    # -------- SHOW WINDOWS --------

    cv2.imshow("ASL Camera + Reference", camera_ui)
    cv2.imshow("ASL AI Dashboard", dashboard)


    key = cv2.waitKey(1) & 0xFF


    # ESC exit
    if key == 27:
        break

    # CAPTURE LETTER
    if key == ord('c'):

        if letter not in ["nothing","del"]:

            if letter == "space":
                current_word += " "
            else:
                current_word += letter

            print("Captured:", letter)


cap.release()
cv2.destroyAllWindows()

print("Program closed")