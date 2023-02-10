# TinyATLANTIS- By: ebgoldstein - Fri Feb 10 2023

import sensor, image, time, lcd, pyb, tf

#setup LEDs and set into known off state
redLED   = pyb.LED(1)
greenLED = pyb.LED(2)
yellowLED  = pyb.LED(3)

sensor.reset() # Initialize the camera sensor.
sensor.set_pixformat(sensor.RGB565) # or sensor.GRAYSCALE
sensor.set_framesize(sensor.QQVGA2) # Special 128x160 framesize for LCD Shield.
sensor.skip_frames(time = 2000)
lcd.init() # Initialize the lcd screen.

#blue light during setup
yellowLED.on()



#Load the TFlite model and the labels
net = tf.load('/ATLANTIS_cat_OMV.tflite', load_to_fb=True)
labels = ['breakwater', 'bridge', 'canal', 'cliff', 'culvert', 'cypress_tree', 'dam', 'ditch',
 'fjord', 'flood', 'glaciers', 'hot_spring', 'lake', 'levee', 'lighthouse', 'mangrove',
 'marsh', 'offshore_platform', 'pier', 'pipeline', 'puddle', 'rapids', 'reservoir', 'river',
 'river_delta', 'sea', 'ship', 'shoreline', 'snow', 'spillway', 'swimming_pool', 'water_tower',
 'water_well', 'waterfall', 'wetland']

#turn blue off when model is loaded
yellowLED.off()


#MAIN LOOP

while(True):

    img = sensor.snapshot()

    #Do the classification and get the object returned by the inference.
    TF_objs = net.classify(img)
    #print(TF_objs)

    #The object has a output, which is a list of classifcation scores
    #for each of the output channels. this model only has 2 (no mokapot, mokapot).
    print(TF_objs[0].output())

    sorted_list = sorted(zip(labels, TF_objs[0].output()), key = lambda x: x[1], reverse = True)
    for i in range(5):
        print("%s = %f" % (sorted_list[i][0], sorted_list[i][1]))

    #state the top 5
    img.draw_string(1,100, sorted_list[0][0], color = (100,10,100), scale = 2,mono_space = False)
    img.draw_string(1,110, sorted_list[1][0], color = (100,10,100), scale = 2,mono_space = False)
    img.draw_string(1,120, sorted_list[2][0], color = (100,10,100), scale = 2,mono_space = False)
    img.draw_string(1,130, sorted_list[3][0], color = (100,10,100), scale = 2,mono_space = False)
    img.draw_string(1,140, sorted_list[4][0], color = (100,10,100), scale = 2,mono_space = False)

    lcd.display(img) # dsiplay image
