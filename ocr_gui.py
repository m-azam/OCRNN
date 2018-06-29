#!/home/bnmit/miniconda3/bin/python3.6
from appJar import gui
import os 
from PIL import Image, ImageTk

app = gui("OCR")
app.setLocation("CENTER")
app.setStretch("both")
app.setPadding([5,5])
app.setSticky("ne")
app.addLabel("userLab", "Image name:", 0, 0,colspan=1)
app.setSticky("nw")
app.addEntry("image_name", 0, 1,colspan=2)

count = 0

def press(btnName):
    pg_count = 0
    pg1_count = 0
    global count
    count += 1
    nam = str(app.getEntry("image_name"))
    if count == 1:
        app.setSticky("nesw")
        app.addMeter("progress", colspan=2)
        app.addMeter("progress1", colspan=2)
        app.setMeterFill("progress", "blue")
        app.setMeterFill("progress1", "blue")
        app.addImage("pic", nam,colspan=2)
    else:
        app.setMeter("progress", 0)
        app.reloadImage("pic", nam)    
    cline = 'python main_r.py '
    app.zoomImage("pic", -2)
    for i in range(50):
        pg1_count+=1
        app.setMeter("progress", pg1_count)
    os.system(cline+ nam)
    for i in range(101):
        pg_count += 1        
        app.setMeter("progress1", pg_count)
        if i>50:
            pg1_count +=1
            app.setMeter("progress", pg1_count)
    app.addLabel("success", "Done!", 5, 0,colspan=2)
    app.setSticky("")
    app.addButton( "Open output file", open_out, 6, 0, colspan=2)
    
    
def open_out(btnName):
    os.system("output_final.txt")
app.setSticky("")
app.addButton( "Submit", press, 1, 0, colspan=2)






app.go()
