#!/usr/bin/env python3
"""Render each dashboard chart to JPEG using Pillow only (matplotlib's Agg SIGILLs on this
no-AVX QEMU CPU). 3x supersample + LANCZOS downscale for anti-aliasing. Data mirrors the site."""
import os
from math import log2
from PIL import Image, ImageDraw, ImageFont

S = 3                      # supersample factor
FTD = "/usr/share/fonts/truetype/dejavu/"
def F(name, sz): return ImageFont.truetype(FTD + name, sz * S)
f_title = F("DejaVuSans-Bold.ttf", 16)
f_lab   = F("DejaVuSans.ttf", 12)
f_leg   = F("DejaVuSans.ttf", 12)
f_mono  = F("DejaVuSansMono.ttf", 11)
f_monob = F("DejaVuSansMono-Bold.ttf", 11)

BLUE=(42,120,214); ORANGE=(235,104,52); AQUA=(27,175,122)
INK=(11,11,11); INK2=(82,81,78); MUTED=(137,135,129)
GRID=(225,224,217); AXIS=(195,194,183); RED=(208,59,59); WHITE=(255,255,255)
os.makedirs("charts", exist_ok=True)

# ---- data ----
DEPTHS=[512,1024,2048,4096,8192,16384,32768,49152,65536,98304,131072,196608,262144]
Q6_TG=[54.78,54.39,54.57,53.58,53.93,53.38,52.01,50.87,49.04,47.64,45.58,42.35,38.32]
Q8_TG=[53.68,53.62,53.51,53.35,52.95,52.21,51.0,49.85,48.43,46.84,44.97,41.17,38.99]
Q6_PP=[1392.4,1311.67,1352.97,1366.59,1354.4,1335.02,1294.3,1252.65,1218.29,1141.86,1080.91,969.43,876.32]
Q8_PP=[1457.79,1336.62,1386.54,1387.67,1383.91,1366.48,1324.1,1280.51,1244.92,1168.3,1104.24,988.81,892.6]
SPEC_X=[496,4109,33021,98991,197095,258903]
S_NONE=[51.92,50.73,48.63,44.76,40.15,37.79]
S_MTP =[92.84,85.83,95.36,78.25,70.36,69.3]
S_NGRAM=[20.92,19.37,18.39,17.01,15.32,17.55]
UB=[256,512,1024,2048,4096]; UB98=[1080,1144,1189,1194,1162]; UB256=[843,875,879,840,833]

def ctx(v):
    v=int(round(v)); return f"{v//1024}K" if v>=1024 else str(v)

# ---- primitive helpers (logical coords, scaled at draw time) ----
def canvas(w,h):
    im=Image.new("RGB",(w*S,h*S),WHITE); return im, ImageDraw.Draw(im)
def line(d,pts,fill,width):
    d.line([(x*S,y*S) for x,y in pts], fill=fill, width=max(1,int(width*S)), joint="curve")
def dash(d,p0,p1,fill,width):
    (x0,y0),(x1,y1)=p0,p1; n=max(2,int(((x1-x0)**2+(y1-y0)**2)**.5/6))
    for i in range(n):
        if i%2: continue
        a=i/n; b=min(1,(i+1)/n)
        d.line([((x0+(x1-x0)*a)*S,(y0+(y1-y0)*a)*S),((x0+(x1-x0)*b)*S,(y0+(y1-y0)*b)*S)],fill=fill,width=max(1,int(width*S)))
def marker(d,x,y,color,r=4.0):
    d.ellipse([(x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S], fill=WHITE, outline=color, width=int(1.8*S))
def rrect(d,x0,y0,x1,y1,fill,rad=4):
    d.rounded_rectangle([x0*S,y0*S,x1*S,y1*S], radius=rad*S, fill=fill)
def T(d,x,y,s,font,fill,anchor="la"):
    d.text((x*S,y*S), s, font=font, fill=fill, anchor=anchor)
def tw(d,s,font): return d.textlength(s,font=font)/S
def save(im,name):
    im.resize((im.width//S, im.height//S), Image.LANCZOS).save("charts/"+name, quality=92)
    print("wrote charts/"+name)

def legend_row(d, items, x, y):
    for name,color in items:
        d.line([(x*S,(y)*S),((x+18)*S,(y)*S)], fill=color, width=int(3*S))
        T(d, x+24, y, name, f_leg, INK2, "lm")
        x += 24 + tw(d,name,f_leg) + 22

def logchart(name, title, ylabel, series, ymax, yticks, xticks, xdom, endlabels):
    w,h=900,470; im,d=canvas(w,h)
    ML,MR,MT,MB=74,72,70,56
    x0,x1,y0,y1=ML,w-MR,MT,h-MB
    lo,hi=log2(xdom[0]),log2(xdom[1])
    X=lambda v: x0+(log2(v)-lo)/(hi-lo)*(x1-x0)
    Y=lambda v: y1-(v/ymax)*(y1-y0)
    T(d, ML, 22, title, f_title, INK, "lm")
    legend_row(d, [(n,c) for (n,c,_) in series], ML, 46)
    for t in yticks:
        yy=Y(t); line(d,[(x0,yy),(x1,yy)],GRID,1); T(d, x0-10, yy, str(t), f_mono, MUTED, "rm")
    for xv in xticks:
        xx=X(xv); T(d, xx, y1+14, ctx(xv), f_mono, MUTED, "ma")
    T(d, x0-10, y0-14, ylabel, f_mono, MUTED, "lm")
    T(d, (x0+x1)/2, h-14, "context (tokens, log scale)", f_lab, MUTED, "mm")
    line(d,[(x0,y1),(x1,y1)],AXIS,1)
    for (nm,color,ys) in series:
        pts=[(X(x),Y(y)) for x,y in zip(SPECX_for(name,ys),ys)]
        line(d,pts,color,2.4)
        for px,py in pts: marker(d,px,py,color)
    for (nm,color,ys),lab in zip(series,endlabels):
        xs=SPECX_for(name,ys); px,py=X(xs[-1]),Y(ys[-1])
        T(d, px+9, py, lab, f_monob, color, "lm")
    save(im,name)

def SPECX_for(name, ys):
    return SPEC_X if len(ys)==6 else DEPTHS

# 1. spec-decode
logchart("01_spec_decode.jpg","Speculative decode — TG vs context  (Q8, tensor, single-user)","decode tok/s",
    [("MTP",ORANGE,S_MTP),("none (baseline)",BLUE,S_NONE),("n-gram",AQUA,S_NGRAM)],
    105,[0,25,50,75,100],[512,4096,32768,131072,262144],(480,300000),["MTP","none","ngram"])
# 2. quant TG
logchart("02_quant_tg.jpg","Decode vs context — Q6_K_XL vs Q8_0  (tensor)","decode tok/s",
    [("Q8_0",ORANGE,Q8_TG),("Q6_K_XL",BLUE,Q6_TG)],
    62,[0,20,40,60],[512,8192,65536,262144],(480,300000),["Q8","Q6"])
# 3. quant PP
logchart("03_quant_pp.jpg","Prefill vs context — Q6_K_XL vs Q8_0  (tensor)","prefill tok/s",
    [("Q8_0",ORANGE,Q8_PP),("Q6_K_XL",BLUE,Q6_PP)],
    1600,[0,400,800,1200,1600],[512,8192,65536,262144],(480,300000),["Q8","Q6"])

# 4. comparison grouped bars
def comparison():
    w,h=900,470; im,d=canvas(w,h); ML,MR,MT,MB=74,30,70,60
    x0,x1,y0,y1=ML,w-MR,MT,h-MB; ymax=1650
    Y=lambda v: y1-(v/ymax)*(y1-y0)
    groups=["Prefill","Decode","Decode + MTP"]; a=[1392,55,95]; b=[1119,40,40]
    T(d,ML,22,"4× RTX 3080  vs  8× RTX 5060 Ti box   (tok/s, matched Q6 / tensor)",f_title,INK,"lm")
    legend_row(d,[("4× RTX 3080",ORANGE),("4× RTX 5060 Ti",BLUE)],ML,46)
    for t in [0,400,800,1200,1600]:
        yy=Y(t); line(d,[(x0,yy),(x1,yy)],GRID,1); T(d,x0-10,yy,str(t),f_mono,MUTED,"rm")
    line(d,[(x0,y1),(x1,y1)],AXIS,1)
    gw=(x1-x0)/len(groups); bw=54
    for i,g in enumerate(groups):
        cx=x0+gw*i+gw/2
        for j,(val,col) in enumerate([(a[i],ORANGE),(b[i],BLUE)]):
            bx=cx+(j-0.5)*(bw+8)-bw/2+bw/2*0; bx=cx-(bw+8)/2+j*(bw+8)
            rrect(d,bx,Y(val),bx+bw,y1,col,rad=4)
            T(d,bx+bw/2,Y(val)-8,str(val),f_monob,INK2,"mm")
        T(d,cx,y1+16,g,f_lab,INK2,"ma")
    save(im,"04_comparison.jpg")
comparison()

# 5. pcie horizontal
def pcie():
    w,h=900,360; im,d=canvas(w,h); ML,MR,MT,MB=210,60,64,46
    x0,x1,y0,y1=ML,w-MR,MT,h-MB; xmax=17.5
    X=lambda v: x0+(v/xmax)*(x1-x0)
    rows=[("decode",0.05,BLUE),("prefill (typical)",3.3,AQUA),("prefill burst (high ctx)",15.5,ORANGE)]
    T(d,30,22,"Intercard PCIe traffic by phase",f_title,INK,"lm")
    legend_row(d,[],30,46)
    for gx in [0,4,8,12,16]:
        xx=X(gx); line(d,[(xx,y0-4),(xx,y1)],GRID,1); T(d,xx,y1+14,str(gx),f_mono,MUTED,"ma")
    T(d,(x0+x1)/2,h-12,"GB/s",f_lab,MUTED,"mm")
    bh=44; gap=(y1-y0-bh*len(rows))/(len(rows)+1)
    for i,(lab,val,col) in enumerate(rows):
        yy=y0+gap+(bh+gap)*i
        rrect(d,x0,yy,X(max(val,0.06)),yy+bh,col,rad=5)
        T(d,x0-14,yy+bh/2,lab,f_lab,INK2,"rm")
        T(d,X(max(val,0.06))+8,yy+bh/2,("~0" if val<0.1 else str(val))+" GB/s",f_monob,col,"lm")
    cx=X(15.75); dash(d,(cx,y0-6),(cx,y1),RED,1.4); T(d,cx-4,y0-14,"Gen3 ×16 ceiling",f_mono,RED,"rm")
    save(im,"05_pcie.jpg")
pcie()

# 6. ubatch linear categorical
def ubatch():
    w,h=900,430; im,d=canvas(w,h); ML,MR,MT,MB=74,60,70,58
    x0,x1,y0,y1=ML,w-MR,MT,h-MB; ymin,ymax=700,1300
    X=lambda i: x0+i/(len(UB)-1)*(x1-x0)
    Y=lambda v: y1-(v-ymin)/(ymax-ymin)*(y1-y0)
    T(d,ML,22,"n_ubatch → prefill throughput   (default 512 already near-optimal)",f_title,INK,"lm")
    legend_row(d,[("98K context",ORANGE),("256K context",BLUE)],ML,46)
    for t in [700,900,1100,1300]:
        yy=Y(t); line(d,[(x0,yy),(x1,yy)],GRID,1); T(d,x0-10,yy,str(t),f_mono,MUTED,"rm")
    for i,u in enumerate(UB): T(d,X(i),y1+14,str(u),f_mono,MUTED,"ma")
    T(d,x0-10,y0-14,"prefill tok/s",f_mono,MUTED,"lm")
    T(d,(x0+x1)/2,h-14,"n_ubatch (forward-pass width, tokens)",f_lab,MUTED,"mm")
    line(d,[(x0,y1),(x1,y1)],AXIS,1)
    for ys,col,lab in [(UB98,ORANGE,"98K"),(UB256,BLUE,"256K")]:
        pts=[(X(i),Y(v)) for i,v in enumerate(ys)]
        line(d,pts,col,2.4)
        for px,py in pts: marker(d,px,py,col)
        T(d,pts[-1][0]+9,pts[-1][1],lab,f_monob,col,"lm")
    save(im,"06_ubatch.jpg")
ubatch()
print("ALL CHARTS DONE")
