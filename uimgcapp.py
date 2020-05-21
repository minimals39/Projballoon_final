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
import numpy as np
from ThreadedCamera import ThreadedCamera
import logging 
"""
Demo program to detect Anomaly from files and webcam

"""

def main():
    # ---===--- Get the filename --- #
    '''filename = sg.popup_get_file('Filename to play')
    if filename:
        cap = cv2.VideoCapture(filename)
        #cap = WebcamVideoStream(src="http://192.168.0.101:8081").start()
    else:
        #vs = WebcamVideoStream(src=0).start()
        cap = cv2.VideoCapture(0)'''
    # ---===--- get stream --- #
    # cap = WebcamVideoStream(src=0).start()
    # cap = cv2.VideoCapture('actions1.mpg')
    # cap = WebcamVideoStream(src="http://192.168.0.100:8081").start()
    # ---===--- initialize --- #
    fps = FPS().start()
    count = 0
    count_th = 20
    rec = False
    sg.theme('Black')
    useVideo = False
    isFirst = True
    now = datetime.now()
    vodname = now.strftime("%m_%d_%Y,%H-%M-%S")

    cap = None
    # ---===--- define the window layout --- #
    tab1_layout = [[sg.T('Battery: '), sg.ProgressBar(1000, orientation='h', size=(10, 10), key='battbar'),sg.Text('Last detected: '),sg.Text(' ',key='-detected-',size=(20, 1)),sg.Button('Capture', key='-STOP-'), sg.Button('EXIT', key='-EXIT-')],
                   [sg.Image(filename='', key='-image-')],
                   [sg.Text('Height: ', size=(15, 1)),
                    sg.Text('', size=(10, 1), justification='center', key='_HEIGHT_')],
                   [sg.Text('Latitude: ', size=(15, 1)),
                    sg.Text('', size=(10, 1), justification='center', key='_LATI_')],
                   [sg.Text('Longitude: ', size=(15, 1)),
                    sg.Text('', size=(10, 1), justification='center', key='_LONGTI_')],
                   [sg.Text('FPS: '), sg.Text(size=(15, 1), key='-FPS-'),],
                   ]

    column1 = [[sg.Text('Detected history', background_color='#333333', justification='center', size=(20, 1)),],
                #sg.Output(size=(40, 20))],
               [sg.Text('Gimbal command: '), sg.Text(size=(15, 1), key='-Gimbal-')]
               ]

    tab2_layout = [[sg.T('cropped/masked')],
                   [sg.Image(filename='', key='-cropped-'), sg.T(' ' * 30), sg.Image(filename='', key='-masked-'),
                    sg.Column(column1, background_color='#333333')],
                   [sg.Text('Size of the object: ', size=(15, 1)), sg.Text('', size=(10, 1), justification='center'),
                    sg.Slider(range=(0, 500), default_value=15, size=(50, 10), orientation='h', key='-slider-')]]

    tab3_layout = [[sg.Image(filename = '' , key='-histp-')],[sg.Text(' ',key ='-history-',size=(20, 1))],[sg.Button('Next'),sg.Button('Refresh'),sg.Button('Prev')]]
 
    layout = [[sg.Text('DemodetectionUI', size=(15, 1), font='Helvetica 20')],
              [sg.TabGroup(
                  [[sg.Tab('Cam view', tab1_layout), 
                  sg.Tab('Area view', tab2_layout), 
                  sg.Tab('Detected view', tab3_layout)]],)],
              ]

    layoutWin2 = [[sg.Text('Tracking and DetectionDemo', key='-STATUS-')],
                  # note must create a layout from scratch every time. No reuse
                  [sg.Button('Start', key='-START-', disabled=True), sg.Button('Connect'), sg.Button('video'),
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
    # subtractor = cv2.createBackgroundSubtractorMOG2(history = 50,varThreshold=50,detectShadows=True)
    object_bounding_box = None
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
    if capp:
        histpic = capp[len(capp)-ptr]

    while True:
        ev1, vals1 = window2.Read(timeout=100)
        if ev1 is None or ev1 == '-START-':
            window2.Close()
            win2_active = False
            if window2['-RECORD-'].Get():
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('./recording/%s.avi' % (vodname), fourcc, 20.0, (1020, 720))
            break
        if ev1 is None or ev1 == 'Exit':
            if out:
                out.release()
            break
        if ev1 is ev1 == 'video':
            filename = sg.popup_get_file('Filename to play')
            useVideo = True
            if filename:
                cap = cv2.VideoCapture(filename)
            else:
                cap = cv2.VideoCapture(0)
        if ev1 is ev1 == 'Connect':
            try:
                s = comm(netip, port)
                s = s.start()
                #cap = cv2.VideoCapture("http://192.168.0.102:8081")
                #streamer = ThreadedCamera("http://192.168.0.102:8081")
                cap = WebcamVideoStream(src="http://192.168.0.102:8081").start()
                time.sleep(1)
            except:
                sg.popup('Error CameraIP not found')

        if not cap:
            window2['-START-'].update(disabled=True)
        else:
            if useVideo:
                captype = 'Video'
            else:
                captype = 'Aerial camera'
            window2['-STATUS-'].update(captype)
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
                frame = lastframe
        else:
            #frame = streamer.grab_frame()
            if frame is not None:
                lastframe = frame.copy()
            try:
                frame = cap.read()
            except:
                frame = None
                while i <= 5 or frame:
                    try:
                        cap = WebcamVideoStream(src="http://192.168.0.102:8081").start()
                        time.sleep(1)
                        frame = cap.read()
                    except:continue


                frame = lastframe
                i = 0


        now = datetime.now()
        s1 = now.strftime("%m_%d_%Y,%H-%M-%S")
        objsize = int(values['-slider-'])
        logging.basicConfig(filename='Detected.log',format='%(asctime)s %(message)s', datefmt='%m_%d_%Y,%H-%M-%S')
        data = "X0_0_0_0"
        if s:
            s.getxyhf()
            data = s.getxyh()
        split = data.split('_')

        if  len(split) > 3 and split[0][0] == 'X':
            window['_LATI_'].update(split[0][1:])
            window['_LONGTI_'].update(split[1])
            window['_HEIGHT_'].update(split[2])
            window['battbar'].update_bar(int(float(split[3][0:2])))


        if frame is None:  # if out of data stop looping
            frame = lastframe
        if out:
            frameout = cv2.resize(frame, (1020, 720))
            out.write(frameout)

        # resizing the big pic
        # frame = cv2.resize(frame,(1280,500))
        # draw a bb around the obj

        # --------Event handler----------------
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
            capp = os.listdir("cap")
            ptr = 1
            if capp:
                histpic = capp[len(capp)-ptr]
                histtemp = cv2.imread("./cap/%s" % (histpic))
                histbytes = cv2.imencode('.png', histtemp)[1].tobytes()  # ditto
                histpic = histpic.replace('-',':')
                window['-histp-'].update(data=histbytes)
                window['-history-'].update(histpic.replace('.jpg',''))

        if event == 'Prev'and capp: 
            if ptr <= len(capp):
                ptr += 1
            histpic = capp[len(capp)-ptr]
            histtemp = cv2.imread("./cap/%s" % (histpic))
            histbytes = cv2.imencode('.png', histtemp)[1].tobytes()  # ditto
            histpic = histpic.replace('-',':')
            window['-histp-'].update(data=histbytes)
            window['-history-'].update(histpic.replace('.jpg',''))


        if event == 'Next'and capp: 
            if ptr > 1:
                ptr -= 1
            histpic = capp[len(capp)-ptr]
            histtemp = cv2.imread("./cap/%s" % (histpic))
            histbytes = cv2.imencode('.png', histtemp)[1].tobytes()  # ditto
            histpic = histpic.replace('-',':')
            window['-histp-'].update(data=histbytes)
            window['-history-'].update(histpic.replace('.jpg',''))




        '''if event == '-detected-':  # stop video
            capp = os.listdir("cap")
            tab3_layout = [[sg.T('Captured')], [sg.Listbox(values=capp,size=(20, 12), key='-LIST-')],
                        [sg.Text(' ',key ='-File-'), sg.Button('Select',size=(15, 1))]]
            winchoose = sg.Window('Detected History', tab3_layout)
            event2, values2 = winchoose.Read()
        if event2 is not None and event2 == 'Select':
            filechoosen = True
            if values2['-LIST-'] and filechoosen == True:
                filechoosen = str(values2['-LIST-'][0])
                winchoose.close()
                histtemp = cv2.imread("./cap/%s" % (filechoosen))
                histbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
                winpic = sg.Window(filechoosen,[[sg.Image(data= histbytes, key='-histp-')],[sg.Button('Exit')]])
                eventp, valuesp = winpic.read()
                winpic['-histp-'].update(data=histbytes)
                if eventp == 'Exit':
                    filechoosen == False
                    winpic.close()'''

            
            # --------bgsubtraction-----------#
        if object_bounding_box is not None:
            success, object_bounding_box = object_tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in object_bounding_box]
                cropped = frame[y:y + h, x:x + w].copy()
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                #detectsize = objsize/2
                cv2.rectangle(frame, (x+objsize, y+objsize), (x + w - objsize, y + h - objsize),(0, 0, 255), 2)
                if (y + (h / 2) - 50) > frame.shape[0] / 2:  # order gimbal
                    if count >= count_th:
                        inp = "u"
                        if s:
                            s.setinp(inp)
                            s.sendcommand()
                        window['-Gimbal-'].update(inp)
                        print((y+h)/2,frame.shape[0])
                        count = 0

                elif (y + (h / 2)) + 50 < frame.shape[0] / 2:
                    if count >= count_th:
                        inp = "d"
                        if s:
                            s.setinp(inp)
                            s.sendcommand()
                        print((y+h)/2,frame.shape[0])

                        window['-Gimbal-'].update(inp)
                        count = 0
                else:
                    if count >= count_th:
                        inp = "s"
                        if s:
                            s.setinp(inp)
                            s.sendcommand()
                        print((y+h)/2,frame.shape[0])

                        window['-Gimbal-'].update(inp)

                        count = 0
                count += 1

        if cropped is not None and cropped.sum() > 0:
            # resizing the small pic
            # cropped = cv2.resize(cropped,(600,400))
            # cv2.imwrite("./all/frame%d.jpg" % count, cropped)
            count += 1
            # mask = subtractor.apply(cropped)
            # for some reason the fast MCD accepts only the %4 size 
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

            # draw contours 
            (cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in cnts:
                if (cv2.contourArea(contour) > objsize):
                    cv2.drawContours(cropped, contour, -1, (0, 255, 0), 2)
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    ##if cv2.contourArea(contour)>50:
                    # print(contour)
                    # cv2.drawContours(cropped,contour,-1,(0,255,0),2)
                    if x <= objsize/2 or x >= cropped.shape[1] - objsize/2 or x + w <= objsize/2 or x + w >= cropped.shape[1] - objsize/2:
                        if fmeter >= 3:
                            cv2.imwrite("./capc/%s.jpg" % (s1), cropped)
                            cv2.imwrite("./cap/%s.jpg" % (s1), frame)
                            logging.warning(" ")
                            fcount += 1
                            window['-detected-'].update(s1)
                            count += 1
                            fmeter = 0
                        else:
                            fmeter += 1
                    elif y <= objsize/2 or y >= cropped.shape[0] - objsize/2 or y + h <= objsize/2 or y + h >= cropped.shape[0] - objsize/2:
                        if fmeter >= 3:
                            cv2.imwrite("./capc/%s.jpg" % (s1), cropped)
                            cv2.imwrite("./cap/%s.jpg" % (s1), frame)
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
        window['-FPS-'].update("{:.2f}".format(fps.fps()))

        if cropped is not None:
            try:
                # print(cropped.shape)
                imgcropped = cv2.imencode('.png', cropped)[1].tobytes()
                imgmarked = cv2.imencode('.png', mask)[1].tobytes()
                cropped_elem.update(data=imgcropped)
                masked_elem.update(data=imgmarked)
            except:
                continue

            #############
            #    | |    #
            #    | |    #
            #    |_|    #
            #  __   __  #
            #  \ \ / /  #
            #   \ V /   #
            #    \_/    #


"""         #############
        # This was another way updates were being done, but seems slower than the above
        img = Image.fromarray(frame)    # create PIL image from frame
        bio = io.BytesIO()              # a binary memory resident stream
        img.save(bio, format= 'PNG')    # save image as png to it
        imgbytes = bio.getvalue()       # this can be used by OpenCV hopefully
        image_elem.update(data=imgbytes)
"""

main()
