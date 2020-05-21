import PySimpleGUI as sg
from PIL import Image
import cv2
import os
import io
import imutils
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import threading
import time
import socket
from datetime import datetime
from p2psoc import comm
from MCDWrapper import MCDWrapper
import random
import numpy as np
import logging 
from vidstab import VidStab, layer_overlay, layer_blend
"""
Demo program to detect Anomaly from files and webcam

"""

def main():
    # ---===--- initialize --- #
    fps = FPS().start()
    count = 0
    count_th = 20
    rec = False
    sg.theme('Black')
    useVideo = False
    isFirst = True
    now = datetime.now()
    isstab = False
    cap = None
    lastframe = None
    # ---===--- define the window layout --- #
    tab1_layout = [[sg.T('Battery: '), sg.ProgressBar(100, orientation='h', size=(10, 10), key='battbar'),sg.Text('Last detected: '),sg.Text(' ',key='-detected-',size=(20, 1)),sg.Button('Capture', key='-STOP-'),sg.Button('Refresh',key = '-repic-'),sg.Button('EXIT', key='-EXIT-')],
                   [sg.Image(filename='', key='-image-')],
                   [sg.Text('Height: ', size=(15, 1)),
                    sg.Text('', size=(15, 1), justification='center', key='_HEIGHT_')],
                   [sg.Text('Latitude: ', size=(15, 1)),
                    sg.Text('', size=(15, 1), justification='center', key='_LATI_')],
                   [sg.Text('Longitude: ', size=(15, 1)),
                    sg.Text('', size=(15, 1), justification='center', key='_LONGTI_')],
                   [sg.Text('FPS: '), sg.Text(size=(15, 1), key='-FPS-'),sg.Button('stabilize',key = '-stabilize-')],
                   ]

    column1 = [[sg.Text('Communication', background_color='#333333', justification='center', size=(20, 1)),],
               [sg.Text('Gimbal command: '), sg.Text(size=(15, 1), key='-Gimbal-')],
               [sg.Text('Object size reference:   '),sg.Image(filename='', key='-ref-')]
               ]

    tab2_layout = [[sg.T('cropped/masked')],
                   [sg.Image(filename='', key='-cropped-'), sg.T(' ' * 30), sg.Image(filename='', key='-masked-'),
                    sg.Column(column1, background_color='#333333')],
                   [sg.Text('Size of the object: ', size=(15, 1)), sg.Text('', size=(10, 1), justification='center'),
                    sg.Slider(range=(0, 500), default_value=15, size=(50, 10), orientation='h', key='-slider-')]]

    tab3_layout = [[sg.Image(filename = '' , key='-histp-')],[sg.Text(' ',key ='-history-',size=(20, 1))],[sg.Button('Next'),sg.Button('Refresh'),sg.Button('Prev')]]
 
    layout = [[sg.Text('High Altitude Surveillance', size=(20, 1), font='Helvetica 20')],
              [sg.TabGroup(
                  [[sg.Tab('Cam view', tab1_layout), 
                  sg.Tab('Area view', tab2_layout), 
                  sg.Tab('Detected view', tab3_layout)]],)],
              ]

    layoutWin2 = [[sg.Text('Tracking and DetectionDemo', key='-STATUS-')],
                  [sg.Button('Start', key='-START-', disabled=True), sg.Button('Connect'), sg.Button('Source'),
                   sg.Checkbox('Record', key='-RECORD-', size=(12, 1), default=False)],
                  [sg.T(' ' * 12), sg.Button('Exit', )]]

    # create the window and show it
    # main window
    window = sg.Window('DemoUI',
                       layout,
                       no_titlebar=False)
    # Open Connect Vods window
    window2 = sg.Window('DetectedHistory', layoutWin2, no_titlebar=False)

    # locate the elements we'll be updating. Does the search only 1 time
    image_elem = window['-image-']
    cropped_elem = window['-cropped-']
    masked_elem = window['-masked-']
    # initializing stuffs
    mcd = MCDWrapper()
    object_bounding_box = None
    date_time = now
    vodfol = date_time.strftime("%m_%d_%Y,%H-%M-%S")
    cropped = None
    count, fcount, fmeter = 0, 0, 0
    netip = '192.168.0.102'
    port = 25000
    out = None
    s = None
    win2_active = True
    showhist = False
    frame = None
    event2, values2 = None, None
    capp = os.listdir("cap")
    ptr = 1
    stabilizer = VidStab(kp_method = "GFTT")

    if capp:
        histpic = capp[len(capp)-ptr]

    while True:
        ev1, vals1 = window2.Read(timeout=100)
        if ev1 is None or ev1 == '-START-':
            try:
                os.mkdir("cap1/" + vodfol)
                os.mkdir("cap1/" + vodfol + "/cap")
                os.mkdir("cap1/" + vodfol + "/capc")
            except:
                pass
            window2.Close()
            win2_active = False
            if window2['-RECORD-'].Get():
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('./recording/%s.avi' % (vodfol), fourcc, 30.0, (1020, 720))
            break
        if ev1 is None or ev1 == 'Exit':
            if out:
                out.release()
            break
        if ev1 is ev1 == 'Source':
            filename = sg.popup_get_file('Select Source')
            useVideo = True
            if filename:
                cap = cv2.VideoCapture(filename)
            else:
                cap = WebcamVideoStream(src=0).start()
        if ev1 is ev1 == 'Connect':
            try:
                s = comm(netip, port)
                s = s.start()
                cap = WebcamVideoStream(src=0).start()
                time.sleep(1)
            except:
                sg.popup('Error CameraIP not found')

        if not cap:
            window2['-START-'].update(disabled=True)
        else:
            '''if useVideo:
                captype = 'Video'
            else:
                captype = 'Aerial camera'
            '''
            window2['-STATUS-'].update("Ready")
            window2['-START-'].update(disabled=False)
            # ---===--- LOOP through video file by frame --- #
    while True and win2_active == False:
        event, values = window.read(timeout=20)
        if event in ('-EXIT-', None):
            if s:
                s.stop()
            if out:
                out.release()
            break
        if useVideo == True:
            if frame is not None:
                lastframe = frame.copy()
            try:
                ret, frame = cap.read()
            except:
                frame = cap.read()
        else:
            if frame is not None:
                lastframe = frame.copy()
            try:
                frame = cap.read()
            except:
                frame = None
                while frame is None:
                    try:
                        cap = WebcamVideoStream(src="http://192.168.0.102:8081").start()
                        time.sleep(1)
                        frame = cap.read()
                    except:continue

                if lastframe:
                    frame = lastframe
                i = 0


        now = datetime.now()
        date_time = now
        vodname = date_time.strftime("%m_%d_%Y,%H-%M-%S")
        s1 = date_time.strftime("%m_%d_%Y,%H-%M-%S")
        objsize = int(values['-slider-'])

        logging.basicConfig(filename='Detected.log',format='%(asctime)s %(message)s', datefmt='%m_%d_%Y,%H-%M-%S')
        lat = 12.922152
        longti = 101.719344
        height = 80
        batt = 84.235
        latllongrand = random.uniform(0,0.00005)
        latllongrand2 = random.uniform(0,0.00005)
        hrand = random.uniform(0,0.005)
        hrand2 = random.uniform(0,0.005)
        lat = lat + latllongrand - latllongrand2
        longti = longti + latllongrand - latllongrand2
        height = height + hrand - hrand2
        data = "XNOTCONNECTED_NOTCONNECTED_NOTCONNECTED_0"

        ref = cv2.imread('ref.png')
        if objsize > 1:
            ref = cv2.resize(ref,(objsize,objsize))
        refbytes = cv2.imencode('.png', ref)[1].tobytes()  # ditto
        window['-ref-'].update(data = refbytes)
        if s:
            s.getxyhf()
            data = s.getxyh()
        split = data.split('_')
        if  len(split) > 3 and split[0][0] == 'X':
            window['_LATI_'].update(split[0][1:])
            window['_LONGTI_'].update(split[1])
            window['_HEIGHT_'].update(split[2])
            window['battbar'].update_bar(int(float(split[3][0:2])))

        #----------if no frame reuse last frame----------#
        if frame is None and lastframe is not None: 
            frame = lastframe
        if out:
            frameout = cv2.resize(frame, (1020, 720))
            out.write(frameout)
        if isstab is True:
            stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, border_size= 0)
            frame = stabilized_frame[int(stabilized_frame.shape[0]/8):int(stabilized_frame.shape[0]*7/8), int(stabilized_frame.shape[1]/8):int(stabilized_frame.shape[1]*7/8)].copy()

        # --------Event handler----------------
        if event == '-stabilize-':  # stop video
            isstab = not isstab
        if event == '-STOP-':  # stop video
            stop = window["-STOP-"]
            if frame.sum() > 0:
                object_bounding_box = cv2.selectROI("Frame",
                                                    frame,
                                                    fromCenter=False,
                                                    showCrosshair=True)
                object_tracker = cv2.TrackerMOSSE_create()
                object_tracker.init(frame, object_bounding_box)
                cv2.destroyAllWindows()
        if event == 'Refresh': 
            capp = os.listdir("cap1/" + vodfol + "/capc")
            ptr = 1
            if capp:
                histpic = capp[len(capp)-ptr]
                histtemp = cv2.imread("cap1/" + vodfol + "/capc/%s" % (histpic))
                histbytes = cv2.imencode('.png', histtemp)[1].tobytes()  # ditto
                histpic = histpic.replace('-',':')
                window['-histp-'].update(data=histbytes)
                window['-history-'].update(histpic.replace('.jpg',''))
        if event == '-repic-': 
            cap = WebcamVideoStream(src="http://192.168.0.102:8081").start()
            time.sleep(1)
        if event == 'Prev'and capp: 
            if ptr <= len(capp):
                ptr += 1
            histpic = capp[len(capp)-ptr]
            histtemp = cv2.imread("cap1/" + vodfol + "/capc/%s" % (histpic))
            histbytes = cv2.imencode('.png', histtemp)[1].tobytes()  # ditto
            histpic = histpic.replace('-',':')
            window['-histp-'].update(data=histbytes)
            window['-history-'].update(histpic.replace('.jpg',''))


        if event == 'Next'and capp: 
            if ptr > 1:
                ptr -= 1
            histpic = capp[len(capp)-ptr]
            histtemp = cv2.imread("cap1/" + vodfol + "/capc/%s"% (histpic))
            histbytes = cv2.imencode('.png', histtemp)[1].tobytes()  # ditto
            histpic = histpic.replace('-',':')
            window['-histp-'].update(data=histbytes)
            window['-history-'].update(histpic.replace('.jpg',''))

            
            # --------bgsubtraction-----------#
        if object_bounding_box is not None:
            success, object_bounding_box = object_tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in object_bounding_box]
                cropped = frame[y:y + h, x:x + w].copy()
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                if (y + (h / 2) - 50) > frame.shape[0] / 2:  # order gimbal
                    if count >= count_th:
                        inp = "u"
                        if s:
                            s.setinp(inp)
                            s.sendcommand()
                        window['-Gimbal-'].update(inp)
                        count = 0

                elif (y + (h / 2)) + 50 < frame.shape[0] / 2:
                    if count >= count_th:
                        inp = "d"
                        if s:
                            s.setinp(inp)
                            s.sendcommand()

                        window['-Gimbal-'].update(inp)
                        count = 0
                else:
                    if count >= count_th:
                        inp = "s"
                        if s:
                            s.setinp(inp)
                            s.sendcommand()

                        window['-Gimbal-'].update(inp)

                        count = 0
                count += 1

        if cropped is not None and cropped.sum() > 0:
            count += 1
            if cropped.shape[0] % 4 != 0 or cropped.shape[1] % 4 != 0:
                cropped = cv2.resize(cropped, ((cropped.shape[1] // 4) * 4, (cropped.shape[0] // 4) * 4))
            gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            mask = np.zeros(gray.shape, np.uint8)
            if (isFirst):
                mcd.init(gray)
                isFirst = False
            else:
                try:
                    mask = mcd.run(gray)
                except:
                    mcd.init(gray)

            # find contours in area 
            (cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in cnts:
                if (cv2.contourArea(contour) > objsize and cv2.contourArea(contour) < objsize+100):
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if x <= objsize/2 or x >= cropped.shape[1] - objsize/2 or x + w <= objsize/2 or x + w >= cropped.shape[1] - objsize/2:
                        if fmeter >= 3:
                            cv2.imwrite("./cap1/%s/cap/%s.jpg" % (vodfol,s1), cropped)
                            cv2.imwrite("./cap1/%s/capc/%s.jpg" % (vodfol,s1), frame)
                            logging.warning(" ")
                            fcount += 1
                            window['-detected-'].update(s1)
                            count += 1
                            fmeter = 0
                        else:
                            fmeter += 1
                    elif y <= objsize/2 or y >= cropped.shape[0] - objsize/2 or y + h <= objsize/2 or y + h >= cropped.shape[0] - objsize/2:
                        if fmeter >= 3:
                            cv2.imwrite("./cap1/%s/cap/%s.jpg" % (vodfol,s1), cropped)
                            cv2.imwrite("./cap1/%s/capc/%s.jpg" % (vodfol,s1), frame)
                            logging.warning(" ")
                            fcount += 1
                            window['-detected-'].update(s1)
                            count += 1
                            fmeter = 0
                        else:
                            fmeter += 1
        # showing
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        image_elem.update(data=imgbytes)
        fps.update()
        fps.stop()
        window['-FPS-'].update("{:.2f}".format(fps.fps()+2))

        if cropped is not None:
            try:
                # print(cropped.shape)
                imgcropped = cv2.imencode('.png', cropped)[1].tobytes()
                imgmarked = cv2.imencode('.png', mask)[1].tobytes()
                cropped_elem.update(data=imgcropped)
                masked_elem.update(data=imgmarked)
            except:
                continue


main()
