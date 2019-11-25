import numpy as np
import sys, time
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
import matplotlib.pyplot as plt
from skimage import io, transform

from PIL import Image
try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

obj_start_index = 58
noobj_start_index = 58
src_start_index = 58

object_name = 'obj3'


def save(key):
    global src_start_index, obj_start_index, noobj_start_index
    # print(event)
    if key == 'o':
        path = './dataset_cup/obj/' + object_name + '/' + object_name + '_' + str(obj_start_index) + '.png'
        obj_start_index += 1 
    elif key == 'n':
        path = './dataset_cup/noobj/no' + object_name + '_' + str(noobj_start_index) + '.png'
        noobj_start_index += 1
        # plt.savefig(path)
    elif key == 'r':
        path = './dataset' + '_' + str(src_start_index) + '.png'
        src_start_index += 1    
    # path = './dataset/obj/' + object_name + '_' + str(start_index) + '.png'

    img = transform.resize(color, (540, 960))
    img = np.fliplr(img)

    io.imsave(path, img)
    # im = Image.fromarray(img).convert('RGB').save(path)
    # image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)

    print('save img', path)

def press(event):
    save(event.key)

# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(FrameType.Color)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())
# object_name = 'img'
# start_index = 1

black = np.zeros((256, 256, 3))
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', press)

start = time.time()

while True:
    frames = listener.waitForNewFrame()

    color = frames["color"].asarray()[:,:,:3]
    color = color[...,::-1]

    # color = transform.resize(color, (540, 960))
    # color = np.fliplr(color)

    plt.clf()
    plt.imshow(color) #Needs to be in row,col order
    plt.pause(0.1)

    listener.release(frames)

device.stop()
device.close()

sys.exit(0)