from PIL import Image
import cv2
import numpy as np

def rgba2rgb( rgba, background=(0, 0 ,0) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

def gen_fake_lasink(imp, dd, lw):

    w, h = imp.size

    rgb = np.array(imp)[:,:,::-1]
    rgb = cv2.resize(rgb, (int(w*dd),int(h*dd)))

    im = np.zeros((int(h*dd),int(w*dd),3)) + 255
    black = np.zeros((int(h*dd),int(w*dd))) +125

    for i in range(0,int(h*dd),4*lw):
            im[i:i+lw,:,0]=0
            im[i:i+lw,:,1]=152
            im[i:i+lw,:,2]=230
            black[i:i+lw,:]=rgb[i:i+lw,:,0]

            im[i+lw:i+2*lw,:,0]=255
            im[i+lw:i+2*lw,:,1]=218
            im[i+lw:i+2*lw,:,2]=0
            black[i+lw:i+2*lw,:]=rgb[i+lw:i+2*lw,:,1]

            im[i+2*lw:i+3*lw,:,0]=229
            im[i+2*lw:i+3*lw,:,1]=39
            im[i+2*lw:i+3*lw,:,2]=130
            black[i+2*lw:i+3*lw,:]=rgb[i+2*lw:i+3*lw,:,2]

            black[i+3*lw:i+4*lw,:]=np.mean(rgb[i+3*lw:i+4*lw,:,:],2)

    im2 = Image.fromarray(im.astype(np.uint8))

    Iblack= Image.fromarray(black.astype('uint8'))

    lasink = im2.copy()
    lasink.putalpha(Iblack)
    res = rgba2rgb(np.array(lasink))
    res = Image.fromarray(res)
    
    return res
