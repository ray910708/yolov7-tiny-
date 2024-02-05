import time
import cv2
import numpy as np
from PIL import Image

import time

from yolo import YOLO

if __name__ == "__main__":
    mode = "video"
    crop            = False
    count           = False
    video_path      = 0
    video_save_path = "img_crop.mp4"
    video_fps       = 30.0
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    yolo = YOLO()

    if mode == "predict":
     while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, 0, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能讀取攝像頭（視頻）")
        timer = 0
        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame1 = Image.fromarray(np.uint8(frame))
            if time.time() - timer > 10:
                timer = time.time()
                frame  = np.array(yolo.detect_image(frame1,1))
            else:
                frame  = np.array(yolo.detect_image(frame1,0))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            frame = cv2.putText(frame, "fps:%.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()