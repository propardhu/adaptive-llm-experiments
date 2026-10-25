"""Editable publication figures; measurements come from the audited JSON."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
S = json.loads((ROOT / "results/summary.json").read_text())
PAPER="#FAF8F3"; INK="#152F3D"; MUTED="#506574"; TEAL="#087E79"
BLUE="#426C99"; ORANGE="#BB522F"; LINE="#DADFD9"; PALEBLUE="#EBF0F7"; PALETEAL="#E5F2EC"
plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none","text.color":INK,
                     "savefig.facecolor":PAPER})
TEXTS=[]

def text(ax,x,y,s,size=22,color=INK,weight="normal",ha="left",va="top",**kw):
    obj=ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,
                linespacing=1.3,**kw)
    TEXTS.append(obj)
    return obj

def base(title,subtitle,tag="EXPLAINED",w=1600,h=1000):
    f,ax=plt.subplots(figsize=(w/100,h/100),dpi=100)
    f.patch.set_facecolor(PAPER)
    f.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis("off")
    text(ax,80,48,"LEARNING DURING USE  /  "+tag,15,TEAL,"bold")
    text(ax,80,101,title,31,INK,"bold")
    text(ax,80,163,subtitle,20,MUTED)
    return f,ax

def footer(ax,note,w=1600,h=1000):
    text(ax,80,h-115,note,16,MUTED)
    ax.plot([80,w-80],[h-61,h-61],color=LINE,lw=1)
    text(ax,80,h-42,"PARTHA SAI GUTTIKONDA",13,INK,"bold")
    text(ax,w-80,h-42,"partha_systems.ai",13,MUTED,ha="right")

def box(ax,x,y,w,h,title,detail,color,fill):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=18",
                facecolor=fill,edgecolor=color,lw=1.5))
    text(ax,x+30,y+24,title,24,color,"bold")
    text(ax,x+30,y+70,detail,19,MUTED)

def arrow(ax,a,b,color=INK):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=20,lw=2.2,color=color))

def save(f,name):
    f.canvas.draw()
    renderer=f.canvas.get_renderer()
    # Text must fit within the export canvas. Intentional labels stay fully visible.
    fb=f.bbox
    for ax in f.axes:
        for t in ax.texts:
            b=t.get_window_extent(renderer)
            assert b.x0>=fb.x0-1 and b.x1<=fb.x1+1 and b.y0>=fb.y0-1 and b.y1<=fb.y1+1,(name,t.get_text(),b)
    f.savefig(OUT/(name+".png"),dpi=160)
    f.savefig(OUT/(name+".svg"))
    plt.close(f)

def mechanism():
    f,ax=base("Two ways to carry an experience forward",
              "The key difference is where the information lives.","THE IDEA")
    ax.plot([800,800],[245,780],color=LINE,lw=1.5)
    text(ax,80,245,"Use saved context",26,BLUE,"bold")
    text(ax,880,245,"Update selected weights",26,TEAL,"bold")
    box(ax,80,315,640,122,"Saved note + new question",
        "Relevant information enters the prompt",BLUE,PALEBLUE)
    box(ax,880,315,640,122,"Selected experience",
        "Examples provide a learning signal",TEAL,PALETEAL)
    arrow(ax,(400,448),(400,518),BLUE)
    arrow(ax,(1200,448),(1200,518),TEAL)
    text(ax,1230,471,"Loss → gradients",18,TEAL,va="center")
    box(ax,80,535,640,122,"Model reads the context",
        "Existing weights stay fixed",BLUE,"white")
    box(ax,880,535,640,122,"Use adapted weights",
        "An optimizer changes selected parameters",TEAL,"white")
    text(ax,80,704,"Keep the note for future prompts",22,INK,"bold")
    text(ax,880,704,"Save the update for later sessions",22,INK,"bold")
    text(ax,80,750,"External memory can persist.",20,MUTED)
    text(ax,880,750,"Adapted parameters can persist.",20,MUTED)
    footer(ax,"These approaches can work together. A weight update does not guarantee better answers or lower cost.")
    save(f,"01-how-learning-works")

def track(ax,x,y,mean,counts,den,color,width=630):
    ax.add_patch(Rectangle((x,y),width,28,facecolor="#E7E9E3",edgecolor="none"))
    ax.add_patch(Rectangle((x,y),width*mean/100,28,facecolor=color,edgecolor="none"))
    lo=min(counts)/den*100;hi=max(counts)/den*100
    xx1=x+width*lo/100;xx2=x+width*hi/100
    ax.plot([xx1,xx2],[y+14,y+14],color=INK,lw=2)
    for xx in [xx1,xx2]:ax.plot([xx,xx],[y+5,y+23],color=INK,lw=2)

def results():
    f,ax=base("Protecting old behavior came with a tradeoff",
              "Matched adapters: the same update recipe, with or without a protection penalty.","THE EXPERIMENT")
    un=S["matched_unprotected"];pr=S["selected"]["internal_adapter_kl"]
    for x,heading,sub,key,den in [
        (80,"New facts","Correct answers to unseen wording","fact_correct",72),
        (880,"Earlier behavior","Agreement on unrelated prompts","retention_agreement",48)
    ]:
        text(ax,x,260,heading,27,INK,"bold")
        text(ax,x,311,sub,20,MUTED)
        for y,rec,col,label in [(395,un,ORANGE,"Without protection"),(535,pr,TEAL,"With protection")]:
            val=np.mean(rec[key])/den*100
            text(ax,x,y,label,21,INK)
            text(ax,x+630,y-9,f"{val:.1f}%",31,col,"bold",ha="right")
            track(ax,x,y+54,val,rec[key],den,col)
        text(ax,x,642,"0%",15,MUTED)
        text(ax,x+315,642,"50%",15,MUTED,ha="center")
        text(ax,x+630,642,"100%",15,MUTED,ha="right")
        text(ax,x,690,"72 probes · 24 fictional facts" if den==72 else "48 probes · Stability, not correctness",18,MUTED)
    ax.plot([80,1520],[755,755],color=LINE,lw=1.5)
    text(ax,80,785,"Frozen baseline: 0% without facts · 63.9% with the relevant fact",22,BLUE,"bold")
    text(ax,80,831,"The context baseline uses perfect fact selection. Retrieval was not evaluated.",17,MUTED)
    footer(ax,"Means and min–max ranges over 3 initializations on the same data. Exact next-token scoring.")
    save(f,"02-the-rewording-test")

def research():
    f,ax=base("Three ways to make learning part of inference",
              "Research differs in what it updates and how it prepares those updates.","THE RESEARCH")
    for x,lab in [(80,"APPROACH"),(515,"WHAT CHANGES"),(1060,"PREPARATION AND RULES")]:
        text(ax,x,260,lab,15,MUTED,"bold")
    rows=[
        (345,"In-Place TTT",TEAL,"Existing feed-forward\nprojection weights","Reported setup includes\ncontinued training"),
        (530,"TTT-E2E",BLUE,"Selected feed-forward\nweights","Meta-training prepares\nthe model to adapt"),
        (715,"Titans",ORANGE,"A dedicated\nneural memory","Learned writing,\nmomentum and forgetting")
    ]
    for y,name,col,where,how in rows:
        text(ax,80,y,name,25,col,"bold")
        text(ax,515,y,where,23,INK,"bold")
        text(ax,1060,y,how,21,MUTED)
        if y<700:ax.plot([80,1520],[y+123,y+123],color=LINE,lw=1)
    footer(ax,"These methods were not reproduced locally. Long-context learning does not itself establish lifelong learning.")
    save(f,"05-ttt-research-map")

def memory():
    f,ax=base("Preparing a memory made its updates more useful",
              "A deliberately simple symbolic task. This is not a language model.","OPTIONAL TECHNICAL NOTE")
    keys=["trained_encoder_no_write","untrained_encoder_with_write","trained_encoder_with_write"]
    labels=["Trained representation\nNo memory write","Random representation\nOne gradient write","Trained representation\nOne gradient write"]
    for x,key,label,color in zip([350,800,1250],keys,labels,["#87959C",BLUE,TEAL]):
        dat=S["synthetic_memory"][key];v=dat["mean_percent"]
        top=700-3.2*v
        ax.add_patch(Rectangle((x-105,top),210,700-top,facecolor=color,edgecolor="none"))
        low=700-3.2*max(dat["percent_by_seed"]);high=700-3.2*min(dat["percent_by_seed"])
        ax.plot([x,x],[low,high],color=INK,lw=2)
        for y in [low,high]:ax.plot([x-12,x+12],[y,y],color=INK,lw=2)
        text(ax,x,top-67,f"{v:.1f}%",36,color,"bold",ha="center")
        text(ax,x,730,label,21,INK,ha="center")
    for v in [0,50,100]:
        y=700-v*3.2
        ax.plot([175,1450],[y,y],color=LINE,lw=1,zorder=0)
        text(ax,145,y,f"{v}%",17,MUTED,ha="right",va="center")
    footer(ax,"Recall of supplied associations · 1,000 unseen assignments per run · 3 runs · Ranges span runs")
    save(f,"03-learning-and-interference")

def weights(ax,x,y,size,changed=False):
    gap=9;cell=(size-3*gap)/4
    for i in range(4):
        for j in range(4):
            col=TEAL if changed and (i,j) in [(0,2),(1,1),(2,3),(3,0)] else "#B8C6D0"
            ax.add_patch(FancyBboxPatch((x+j*(cell+gap),y+i*(cell+gap)),cell,cell,
                         boxstyle="round,pad=0,rounding_size=5",facecolor=col,edgecolor="none"))

def social():
    f,ax=plt.subplots(figsize=(10.8,13.5),dpi=100)
    f.patch.set_facecolor(PAPER);f.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set_xlim(0,1080);ax.set_ylim(1350,0);ax.axis("off")
    text(ax,75,65,"LEARNING DURING USE",15,TEAL,"bold")
    text(ax,75,135,"What if LLMs\ncould learn while\nyou use them?",42,INK,"bold")
    text(ax,75,390,"A small experiment in changing weights\nand keeping what already works.",22,MUTED)
    weights(ax,150,590,210,False);weights(ax,705,590,210,True)
    arrow(ax,(410,695),(650,695),TEAL)
    text(ax,530,602,"Experience",20,TEAL,"bold",ha="center")
    text(ax,530,735,"Gradient update",17,MUTED,ha="center")
    text(ax,255,835,"Before",23,INK,"bold",ha="center")
    text(ax,810,835,"Selected weights change",21,INK,"bold",ha="center")
    text(ax,75,970,"Does the next answer improve?",28,INK,"bold")
    text(ax,75,1035,"Our results reveal a tradeoff between\nlearning new facts and preserving behavior.",22,MUTED)
    text(ax,75,1185,"Illustration of selective updates · Results and limits in the article",14,MUTED)
    ax.plot([75,1005],[1255,1255],color=LINE,lw=1)
    text(ax,75,1282,"PARTHA SAI GUTTIKONDA",12,INK,"bold")
    text(ax,1005,1282,"partha_systems.ai",12,MUTED,ha="right")
    save(f,"04-social-card")

if __name__=="__main__":
    mechanism();results();research();memory();social()
    print("Five diagrams and charts exported as PNG and editable SVG.")
