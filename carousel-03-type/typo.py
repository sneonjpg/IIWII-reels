from PIL import Image,ImageDraw,ImageFont,ImageFilter
import numpy as np
W,H=1080,1350
LORA='/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf'
def F(s,wt=700):
    f=ImageFont.truetype(LORA,s)
    try: f.set_variation_by_axes([wt])
    except Exception: pass
    return f
INK=(27,27,29);PAPER=(226,220,212);ORG=(214,98,66)
rng=np.random.default_rng(3)
def bg():
    a=np.ones((H,W,3))*np.array(PAPER)
    n=np.asarray(Image.fromarray(rng.normal(0,1,(H//6,W//6)).astype('float32')).resize((W,H),Image.BICUBIC))
    a+=n[...,None]*5
    a+=rng.normal(0,5,(H,W,1))
    yy,xx=np.mgrid[0:H,0:W];v=((xx-W/2)**2/(W*.8)**2+(yy-H/2)**2/(H*.8)**2)
    a-=v[...,None]*22
    return Image.fromarray(a.clip(0,255).astype('uint8'))
def centered(d,txt,y,font,fill,x=W//2):
    w=d.textlength(txt,font=font);d.text((x-w/2,y),txt,font=font,fill=fill);return w
def logo(im):
    lg=Image.open('IIWII-logo-transparent.png').convert('RGBA');w=190;lg=lg.resize((w,int(lg.height*w/lg.width)),Image.LANCZOS)
    im.paste(lg,((W-w)//2,H-lg.height-58),lg)
def slide(n):
    im=bg();d=ImageDraw.Draw(im)
    if n==1:
        centered(d,'Balance',250,F(150),INK)
        centered(d,"isn't",410,F(120,500),INK)
        # uneven 50/50: left 50 bigger weight, right 50 smaller, shown on a tilted bar
        d.text((120,620),'50',font=F(330,800),fill=INK)
        d.text((700,780),'50',font=F(200,500),fill=INK)
        d.text((520,640),'/',font=F(300,400),fill=ORG)
    if n==2:
        centered(d,"It's what",260,F(120,500),INK)
        centered(d,'today',390,F(300,800),ORG)
        centered(d,'actually needs.',730,F(110,500),INK)
    if n==3:
        centered(d,'Some days',280,F(100,500),INK)
        centered(d,"that's",390,F(100,500),INK)
        centered(d,'WORK.',520,F(300,800),INK)
        centered(d,'rest.',930,F(80,400),(150,144,136))
    if n==4:
        centered(d,'work.',230,F(80,400),(150,144,136))
        centered(d,'Some days',420,F(100,500),INK)
        centered(d,"that's",530,F(100,500),INK)
        centered(d,'rest.',650,F(300,800),ORG)
    if n==5:
        centered(d,'What does',250,F(130,500),INK)
        centered(d,'today',390,F(300,800),ORG)
        centered(d,'need?',730,F(190,700),INK)
        centered(d,'Save this.',1000,F(80,500),INK)
    d.rectangle((80,1160,W-80,1162),fill=(200,194,186)) if False else None
    logo(im);return im
outs=[slide(i) for i in range(1,6)]
for i,im in enumerate(outs):im.convert('RGB').save(f'typo-{i+1}.jpg',quality=93)
s=Image.new('RGB',(2000,500))
for i,im in enumerate(outs):s.paste(im.resize((400,500)),(i*400,0))
s.save('typo-strip.png')
