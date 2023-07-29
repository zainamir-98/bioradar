# Use this command if numpy import fails: sudo apt-get install python-dev libatlas-base-dev
# If this doesn't work, uninstall both numpy and scipy. Thonny will keep an older default version of numpy.
# Install an older version of scipy that corresponds to the correct version of numpy.

from guizero import App, PushButton, Slider,  Text, ButtonGroup, Picture, Box, CheckBox
import sys
import time
import subprocess
import os

DEBUG_MODE = False
#CONT_REALTIME_MONITORING = False

def gui_open_rr_hr():
    app.destroy()
    #os.system("cmd /c py final.py -u")
    process = subprocess.run('python3 scripts/run_rr_hr.py -u', shell=True)
    
def gui_open_hrv_hr():
    app.destroy()
    process = subprocess.run('python3 scripts/run_hrv_hr.py -u', shell=True)

def gui_go_to_connect():
    print("Connecting...")
    start_menu_box.hide()
    connect_menu_box.show()
    
    start_footer_box.hide()
    other_footer_box.show()
    
    connect_menu_text2.hide()
    # Connection function
    connect_menu_text.after(1000, gui_check_connection)
    
def gui_go_to_manual():
    start_menu_box.hide()
    manual_menu_box.show()
    
    start_footer_box.hide()
    other_footer_box.show()
    
def gui_check_connection():
    app.destroy()
    process = subprocess.run('python3 scripts/online_test.py -u', shell=True)
    
def gui_go_back_to_menu():
    connect_menu_box.hide()
    manual_menu_box.hide()
    if connect_menu_text.value == "Connected!":
        connect_menu_text.value = "Connecting to MyVitals..."
    
    start_menu_box.show()
    other_footer_box.hide()
    start_footer_box.show()
    

app = App(title="BioRadar (Prototype)", width=480, height=320, bg="#141414")
if not DEBUG_MODE:
    app.full_screen = True

start_menu_box = Box(app, width="fill")
pad_1 = Box(start_menu_box, width="fill", height=20)
box_1 = Box(start_menu_box, width="fill")
pad_1_2 = Box(box_1, width=140, height=1, align="left")
picture = Picture(box_1, image="images/brlogo.png", width=51, height=40, align="left") # W:H = 1.277
pad_1_2 = Box(box_1, width=10, height=1, align="left")
message = Text(box_1, text="BioRadar", color="#FFFFFF", size=20, align="left")
pad_2 = Box(start_menu_box, width="fill", height=40)
message = Text(start_menu_box, text="Select how you want to monitor your vitals.", color="#FFFFFF", size=15)
pad_3 = Box(start_menu_box, width="fill", height=18)
button1 = PushButton(start_menu_box, text="Online mode", command=gui_go_to_connect)
button1.bg = "#6ED3A9"
pad_4 = Box(start_menu_box, width="fill", height=10)
button2 = PushButton(start_menu_box, text="Manual mode", command=gui_go_to_manual)
button2.bg = "#6ED3A9"

connect_menu_box = Box(app, width="fill")
pad_1 = Box(connect_menu_box, width="fill", height=100)
connect_menu_text = Text(connect_menu_box, text="Connecting to MyVitals...", color="#FFFFFF", size=20)
pad_2 = Box(connect_menu_box, width="fill", height=30)
connect_menu_text2 = Text(connect_menu_box, text="Waiting for online commands...", color="#FFFFFF", size=16)
connect_menu_box.hide()

# Manual mode

manual_menu_box = Box(app, width="fill")
pad = Box(manual_menu_box, width="fill", height=20)
manual_menu_text = Text(manual_menu_box, text="Manual Mode", color="#FFFFFF", size=20)
pad = Box(manual_menu_box, width="fill", height=50)
button_box = Box(manual_menu_box, width=460, height=90)
button1 = PushButton(button_box, text="Respiration Rate\nHeart Rate", command=gui_open_rr_hr, align="left")
pad = Box(button_box, width=10, height=90, align="left")
button2 = PushButton(button_box, text="Heart Rate Variability\nHeart Rate*", command=gui_open_hrv_hr, align="right")
button1.text_size = 16
button2.text_size = 16
button1.bg = "#6ED3A9"
button2.bg = "#6ED3A9"
pad = Box(manual_menu_box, width="fill", height=30)
pad = Box(manual_menu_box, width="fill", height=6)
txt = Text(manual_menu_box, text="* You will need to hold your breath for 10 seconds for\nheart rate variability measurements.", color="#C8C8C8", size=11)
manual_menu_box.hide()

# Footers

start_footer_box = Box(app, width="fill", align="bottom")
fyp_text = Text(start_footer_box, text="   © 2021 Final-Year Project, SEECS, NUST", color="#C8C8C8", size=11, align="left")
exit_button = PushButton(start_footer_box, text="Exit", align="right", command=exit)
exit_button.bg = "#6ED3A9"

other_footer_box = Box(app, width="fill", align="bottom")
exit_button = PushButton(other_footer_box, text="Exit", align="right", command=exit)
exit_button.bg = "#6ED3A9"
back_button = PushButton(other_footer_box, text="Back", align="right", command=gui_go_back_to_menu)
back_button.bg = "#6ED3A9"
other_footer_box.hide()
             
app.display()
