#import py auto guy
import pyautogui
import time

# get the screen size
screenWidth, screenHeight = pyautogui.size()
time.sleep(6)
# get the current mouse position
currentMouseX, currentMouseY = pyautogui.position()

print(currentMouseX, currentMouseY)