#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_studio_hdrs.py
批量生成多种studio风格HDR环境图 (.hdr)
支持左右/上下/对角/三扇/五扇形布局，内含多种配色方案。
依赖：numpy、opencv-python
"""

import argparse, os, random, math, re
import numpy as np, cv2

# ===== 色彩处理 =====

def srgb_to_linear(c):
    a = 0.055
    low = c <= 0.04045
    high = ~low
    out = np.empty_like(c, dtype=np.float32)
    out[low] = (c[low] / 12.92)
    out[high] = (((c[high] + a) / (1 + a)) ** 2.4)
    return out.astype(np.float32)

def hex_to_srgb(hex_str):
    hex_str = hex_str.strip().lstrip('#')
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return np.array([r,g,b], dtype=np.float32)

def kelvin_to_srgb(T):
    T = float(T)
    T = np.clip(T, 1000.0, 40000.0) / 100.0
    if T <= 66:
        R = 255
    else:
        R = 329.698727446 * ((T - 60) ** -0.1332047592)
        R = np.clip(R, 0, 255)
    if T <= 66:
        G = 99.4708025861 * math.log(T) - 161.1195681661
    else:
        G = 288.1221695283 * ((T - 60) ** -0.0755148492)
    G = np.clip(G, 0, 255)
    if T >= 66:
        B = 255
    else:
        if T <= 19:
            B = 0
        else:
            B = 138.5177312231 * math.log(T - 10) - 305.0447927307
            B = np.clip(B, 0, 255)
    return np.array([R, G, B], dtype=np.float32) / 255.0

# ===== 布局生成函数 =====

def make_split_lr(h,w,c1,c2,blend=0.3):
    img=np.zeros((h,w,3),np.float32)
    mid=w//2; b=int(w*blend/2)
    img[:,:mid-b]=c1; img[:,mid+b:]=c2
    if b>0:
        t=np.linspace(0,1,2*b).reshape(1,-1,1)
        img[:,mid-b:mid+b]=c1*(1-t)+c2*t
    return img

def make_split_ud(h,w,c1,c2,blend=0.3):
    img=np.zeros((h,w,3),np.float32)
    mid=h//2; b=int(h*blend/2)
    img[:mid-b]=c1; img[mid+b:]=c2
    if b>0:
        t=np.linspace(0,1,2*b).reshape(-1,1,1)
        img[mid-b:mid+b]=c1*(1-t)+c2*t
    return img

def make_split_diag(h,w,c1,c2,blend=0.3,tilt=45):
    img=np.zeros((h,w,3),np.float32)
    yy,xx=np.mgrid[0:h,0:w]
    angle=np.deg2rad(tilt)
    d=(xx*np.cos(angle)+yy*np.sin(angle))
    d=d/np.max(d)
    t=(d-d.min())/(d.max()-d.min())
    blendw=blend*0.5
    mask=np.clip((t-0.5)/blendw+0.5,0,1)
    img=c1*(1-mask[:,:,None])+c2*mask[:,:,None]
    return img

def make_fan(h,w,colors):
    """生成多扇形布局 (len(colors) >=3)"""
    cx,cy=w/2,h/2
    yy,xx=np.mgrid[0:h,0:w]
    ang=(np.degrees(np.arctan2(yy-cy,xx-cx))+360)%360
    img=np.zeros((h,w,3),np.float32)
    step=360/len(colors)
    for i,c in enumerate(colors):
        m=(ang>=i*step)&(ang<(i+1)*step)
        img[m]=c
    return img

# ===== 主生成逻辑 =====

def make_hdr(w,h,mode,colors):
    if mode=='split_lr': return make_split_lr(h,w,colors[0],colors[1])
    if mode=='split_ud': return make_split_ud(h,w,colors[0],colors[1])
    if mode=='split_diag1': return make_split_diag(h,w,colors[0],colors[1],tilt=45)
    if mode=='split_diag2': return make_split_diag(h,w,colors[0],colors[1],tilt=135)
    if mode=='tri_fan': return make_fan(h,w,colors[:3])
    if mode=='penta_fan': return make_fan(h,w,colors[:5])
    raise ValueError('未知模式 '+mode)

def palette(name):
    if name=='warmcool': return [np.array([0.25,0.4,1.0]), np.array([1.0,0.5,0.2])]
    if name=='magenta_teal': return [np.array([1.0,0.2,0.9]), np.array([0.0,0.8,0.7])]
    if name=='cyan_yellow': return [np.array([0.0,0.9,1.0]), np.array([1.0,1.0,0.2])]
    if name=='red_blue': return [np.array([1.0,0.1,0.1]), np.array([0.1,0.3,1.0])]
    if name=='studio5':
        return [np.array([1.0,0.3,0.3]),np.array([0.3,0.5,1.0]),np.array([0.2,1.0,0.8]),np.array([1.0,0.8,0.3]),np.array([0.9,0.3,1.0])]
    if name=='random':
        k=random.randint(3,6)
        hsv=np.stack([np.linspace(0,1,k,endpoint=False),np.ones(k),np.ones(k)],-1)
        rgb=[np.array(cv2.cvtColor(np.uint8([[c*255]]),cv2.COLOR_HSV2RGB)[0,0])/255. for c in hsv]
        return rgb
    raise ValueError('未知调色 '+name)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out_dir',default='./out_hdrs')
    ap.add_argument('--count',type=int,default=400)
    ap.add_argument('--width',type=int,default=2048)
    ap.add_argument('--height',type=int,default=1024)
    ap.add_argument('--preview',action='store_true')
    args=ap.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)

    modes=['split_lr','split_ud','split_diag1','split_diag2','tri_fan','penta_fan']
    pals=['warmcool','magenta_teal','cyan_yellow','red_blue','studio5','random']

    for i in range(args.count):
        mode=random.choice(modes)
        pal=random.choice(pals)
        cols=palette(pal)
        img=make_hdr(args.width,args.height,mode,cols)
        img*=2.0**random.uniform(-0.5,1.5)  # 随机曝光
        name=f"studio_{i:03d}_{mode}_{pal}.hdr"
        out=os.path.join(args.out_dir,name)
        ok=cv2.imwrite(out,img[...,::-1].astype(np.float32))
        if ok:
            print(f"[OK] {name}")
        if args.preview:
            prev=(np.clip(img,0,1)**(1/2.2))*255
            cv2.imwrite(out.replace('.hdr','_preview.png'),prev[...,::-1].astype(np.uint8))

if __name__=="__main__":
    main()
