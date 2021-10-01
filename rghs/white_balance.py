import numpy as np
import matplotlib.pyplot as plt
import cv2

def white_balance_1(img,lamda=0.1,percentile=3):
    # im = img*255
    # im = im.astype(np.uint8)
    # im = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    # print(np.amax(img,axis=(0,1)))
    im = img.astype("float32")/255

# algo-1
    # ilum = im.mean(axis=(0,1))
    # ilum = ilum[1]/ilum
    # im = im/ilum
    # ilum = 0.5+lamda*ilum
    # im = (im*ilum).clip(0,1)

# algo-2
    r,g,b = (im[:,:,0],im[:,:,1],im[:,:,2])
    rrc = g*(1-r)
    r = r + rrc*(np.mean(g-r))
    rbc = g*(1-b)
    b = b + rbc*(np.mean(g-b))
    im[:,:,0] = r
    im[:,:,2] = b
    im = im*255
    im = im.astype(np.uint8)
    for i,channel in enumerate(cv2.split(im)):
        mi, ma = (np.percentile(channel, percentile), np.percentile(channel,100.0-percentile))
        c = channel.clip(mi,ma)
        c = (c-mi)*255.0/(ma-mi)
        im[:,:,i] = np.uint8(np.clip(c, 0, 255))
    im = im.astype("float32")/255

# algo-3
    # r,g,b = (im[:,:,0],im[0,0,1],im[0,0,2])
    # rrc = g*(1-r)
    # r = r + rrc*(np.mean(g-r))
    # rbc = g*(1-b)
    # b = b + rbc*(np.mean(g-b))
    # im[:,:,0] = r
    # im[:,:,2] = b
    ilum = np.mean(im,axis=(0,1))
    ilum = 0.5+lamda/ilum
    im = im*ilum
    im = im.clip(0,1)

    im = im*255
    im = im.astype(np.uint8)
    return im


def color_correction(img,eta=1):
    im = img.copy()
    g = img[:,:,1]
    im_avg = np.mean(im,axis=(0,1))
    w1 = np.zeros(im.shape)
    for i in range(3):
        w = im[:,:,i] + eta*(im_avg[1]-im_avg[i])*(1-im[:,:,i])*g
        w1[:,:,i] = w   
    return w1

def streching(w1,per=5):
    a = np.percentile(w1,per,axis=(0,1))
    b = np.percentile(w1,100-per,axis=(0,1))
    c = np.amax(w1,axis=(0,1))
    d = np.amin(w1,axis=(0,1))
    w2 = np.zeros(w1.shape)
    for i in range(3):
        w2[:,:,i] = (w1[:,:,i]-a[i])*((c[i]-d[i])/(b[i]-a[i])) + d[i]
    return w2

def gray_world(w2):
    gray = np.sum(w2)/3
    p = gray/np.sum(w2,axis=(0,1))
    w3 = np.zeros(w2.shape)
    for i in range(3):
        w3[:,:,i] = p[i]*w2[:,:,i]
    w = w3.clip(0,1)
    return w

def uwb(img,eta=1,per=5):
    img = img.astype("float32")/255
    w1 = color_correction(img,eta)
    w2 = streching(w1,per)
    w = gray_world(w2)
    w = w*255
    w = w.astype(np.uint8)
    return w