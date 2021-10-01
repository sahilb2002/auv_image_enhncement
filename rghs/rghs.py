import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

def show(imgs,titles=None):
    n=len(imgs)
    plt.figure(figsize=(30,20))
    for i in range(n):
        plt.subplot(2,(n+1)//2,i+1)
        plt.imshow(imgs[i])
        plt.axis("off")
        if titles != None:
            plt.title(titles[i])
    plt.show()




def gray_world2(img):
    # img is uint 0-255 rgb image.
    im = img.astype("float32")
    g_avg = np.mean(im[:,:,1])/255
    b_avg = np.mean(im[:,:,2])/255
    t = np.array([1.0,0.5/g_avg,0.5/b_avg])
    out = np.zeros(im.shape)
    for i in range(3):
        out[:,:,i] = t[i]*im[:,:,i]
    out = out.astype(np.uint)
    return out

def hist_mode_index(c):
    # c is single channel of uint 0-255 image
    # returns histogram,mode, index of mode of c
    flat = c.flatten()
    out = np.bincount(flat,minlength=256)
    mode = np.argmax(out)
    # print(out)
    if mode==0:
        mode = 1
    mode_index = np.sum(out[0:mode])+1
    return (out,mode,mode_index)

def find_min_max(c,tl,kappa,per=0.1):
    # c is single channel of uint 0-255 image
    hist,mode,index = hist_mode_index(c)
    ind = int(index*per/100)
    n = len(hist)
    su=0
    imin=0
    for i in range(n):
        su+=hist[i]
        if su>=ind:
            imin=i
            break
    su=0
    imax=0
    for i in np.arange(n-1,-1,-1):
        su += hist[i]
        if su>=ind:
            imax=i
            break
    sigma = np.sqrt((4-np.pi)/2)*mode
    # print(sigma,mode)
    omin = int(mode - 1.5*sigma)
    omax=0
    ll = kappa*tl*imax/sigma - 1.526
    ul = kappa*tl*255/sigma - 1.526
    sols = np.arange(np.ceil(ll),np.floor(ul)+1,1)
    if len(sols)<1:
        omax=255
    else:
        mu = np.mean(sols)
        omax = int((mode+mu*sigma)/(kappa*tl))
    return (imin,imax,omin,omax)

def kappa_tl(i,d=3):
    kap=0.9
    tl=0
    if i==0:
        kap=1.1
        tl = 0.83**d
    elif i==1:
        tl = 0.95**d
    else:
        tl = 0.97**d
    return (kap,tl)
    


def rghs(image,per=0.1):
    # img is uint 0-255 rgb image.
    img = image.astype(np.int32)
    # print(np.amax(img,axis=(0,1)))
    out = np.zeros(img.shape)
    im = img.astype("float32")
    for i in range(3):
        kappa,tl = kappa_tl(i)
        imin,imax,omin,omax = find_min_max(img[:,:,i],tl,kappa,per)
        out[:,:,i] = (im[:,:,i]-imin)*((omax-omin)/(imax-imin)) + omin
    out = out.astype(np.uint8)
    return out


def color_correction(img,per=0.1):

    img = img.astype(np.uint8)
    im = rgb2lab(img)
    out = np.zeros(im.shape)

    mi = np.percentile(im[:,:,0],per)
    ma = np.percentile(im[:,:,0],100-per)
    im[:,:,0] = im[:,:,0].clip(mi,ma)
    out[:,:,0] = (im[:,:,0]-mi)*(100-0)/(ma-mi)

    for i in [1,2]:
        out[:,:,i] = im[:,:,i]*(1.3**(1-np.abs(im[:,:,i]/128)))

    out2 = lab2rgb(out)*255
    out2 = out2.astype(np.uint)
    return out2

def enhance_rghs(img):
    color_eq = gray_world2(img)
    rghs_img = rghs(color_eq)
    result = color_correction(rghs_img)
    return result