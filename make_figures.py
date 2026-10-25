"""Exact diagrams and charts from audited study results; PNG and editable SVG."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parent
S=json.loads((ROOT/'results/results.json').read_text())
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
PAPER='#FAF8F3';INK='#17313D';MUTED='#526977';TEAL='#087F79';BLUE='#426E9A';ORANGE='#BD532F';LINE='#DADFD9'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','text.color':INK,'savefig.facecolor':PAPER})

def txt(ax,x,y,s,size=21,color=INK,weight='normal',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,weight=weight,va='top',linespacing=1.25,**kw)

def base(title,subtitle,tag):
    f,ax=plt.subplots(figsize=(16,10),dpi=100)
    f.subplots_adjust(0,0,1,1);f.patch.set_facecolor(PAPER)
    ax.set_xlim(0,1600);ax.set_ylim(1000,0);ax.axis('off')
    txt(ax,70,42,'TITANS INSPIRED MEMORY  /  '+tag,14,TEAL,'bold')
    txt(ax,70,92,title,30,INK,'bold')
    txt(ax,70,154,subtitle,19,MUTED)
    return f,ax

def footer(ax,line):
    txt(ax,70,883,line,15,MUTED)
    ax.plot([70,1530],[941,941],color=LINE,lw=1)
    txt(ax,70,961,'PARTHA SAI GUTTIKONDA',12,INK,'bold')
    txt(ax,1530,961,'partha_systems.ai',12,MUTED,ha='right')

def box(ax,x,y,w,h,title,body,color,fill):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=16',facecolor=fill,edgecolor=color,lw=1.5))
    txt(ax,x+26,y+23,title,23,color,'bold');txt(ax,x+26,y+76,body,19,MUTED)

def arrow(ax,a,b,color=INK):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=22,lw=2.2,color=color))

def save(f,name):
    f.canvas.draw();renderer=f.canvas.get_renderer();bounds=f.bbox
    for ax in f.axes:
        for t in ax.texts:
            b=t.get_window_extent(renderer)
            assert b.x0>=bounds.x0-1 and b.x1<=bounds.x1+1 and b.y0>=bounds.y0-1 and b.y1<=bounds.y1+1,(name,t.get_text(),b)
    f.savefig(OUT/(name+'.png'),dpi=150)
    f.savefig(OUT/(name+'.svg'))
    svg_path=OUT/(name+'.svg')
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
    plt.close(f)

def architecture():
    f,ax=base('Two parts with different jobs',
              'One part stays fixed. A separate memory learns as information arrives.','THE IDEA')
    box(ax,70,255,585,215,'The part that stays fixed',
        'Prepares incoming information\nGuides the memory’s updates\nStays unchanged during use',BLUE,'#EBF0F7')
    box(ax,945,255,585,215,'The memory that learns',
        'Stores associations in its own weights\nChanges as new information arrives\nCan be saved or reset',TEAL,'#E5F2EC')
    arrow(ax,(668,365),(931,365),TEAL)
    txt(ax,800,274,'Information\nto remember',18,MUTED,ha='center')
    txt(ax,800,390,'Small adjustments\nto memory weights',17,TEAL,ha='center')
    arrow(ax,(1235,480),(1235,550),TEAL)
    box(ax,945,565,585,145,'Recall what was learned',
        'Use a learned association to answer',TEAL,'white')
    box(ax,70,565,585,145,'Choose what to keep',
        'Save, reload or reset the memory\nRebuild it from trusted information',BLUE,'white')
    txt(ax,70,770,'A separate memory still learns into weights.',27,INK,'bold')
    txt(ax,70,820,'Weights are the internal numbers that determine what the memory recalls.',20,MUTED)
    footer(ax,'This shows the small memory system we tested. It is not a complete AI assistant or a full Titans model.')
    save(f,'01-where-learning-happens')

def results():
    f,ax=base('The memory learned mistakes too',
              'A small memory test with 32 artificial pairings and eight possible answers.','THE EXPERIMENT')
    rows=[('Before storing the facts','no_write',MUTED),
          ('After learning correct pairs','clean_memory',TEAL),
          ('After 8 wrong updates','after_false_updates',ORANGE),
          ('After correcting those 8','after_verified_corrections',BLUE),
          ('After reset and clean replay','reset_and_replay',TEAL)]
    for i,(label,key,color) in enumerate(rows):
        y=270+i*90;v=S['summary']['conditions'][key]
        txt(ax,70,y,label,20,INK)
        x,w=610,755
        ax.barh(y+15,w,height=29,left=x,color='#E5E9E2')
        ax.barh(y+15,w*v['accuracy'],height=29,left=x,color=color)
        lo=x+w*v['min'];hi=x+w*v['max']
        ax.plot([lo,hi],[y+15,y+15],color=INK,lw=1.8)
        for xx in (lo,hi):ax.plot([xx,xx],[y+6,y+24],color=INK,lw=1.8)
        txt(ax,1500,y-2,f"{100*v['accuracy']:.1f}%",25,color,'bold',ha='right')
    for val in [0,.5,1]:txt(ax,610+755*val,746,f'{val*100:.0f}%',14,MUTED,ha='center')
    txt(ax,70,801,'Only the separate memory changed during each test.',22,INK,'bold')
    footer(ax,'Three trained versions faced the same 200 new test sets. Thin marks show variation between versions.\nA simple lookup table scored 100% on clean pairs. This was not a full language-model test.')
    save(f,'02-learning-and-false-updates')

def capacity():
    f,ax=base('Recall fell as memory load increased',
              'The same 2,560 memory weights must store more associations as the stream grows.','CAPACITY AND CONTROL')
    chart=f.add_axes([.12,.32,.77,.44],facecolor=PAPER)
    loads=np.array(S['protocol']['memory_loads'])
    means=np.array([S['summary']['capacity'][str(n)] for n in loads])*100
    low=np.array([min(r['capacity'][str(n)]['accuracy'] for r in S['runs']) for n in loads])*100
    high=np.array([max(r['capacity'][str(n)]['accuracy'] for r in S['runs']) for n in loads])*100
    chart.axhline(100,color=BLUE,lw=1.5,ls='--',label='Exact-key lookup')
    chart.errorbar(loads,means,yerr=[means-low,high-means],color=TEAL,lw=3,marker='o',markersize=9,capsize=5,label='Learned neural memory')
    chart.set_ylim(45,108);chart.set_xlim(4,70);chart.set_xticks(loads)
    chart.set_yticks([50,75,100]);chart.set_yticklabels(['50%','75%','100%'])
    chart.tick_params(labelsize=16,colors=MUTED);chart.grid(axis='y',color=LINE)
    chart.set_xlabel('Associations written before recall',fontsize=17,labelpad=14,color=MUTED)
    chart.set_ylabel('Exact recall',fontsize=17,labelpad=14,color=MUTED)
    for spine in chart.spines.values():spine.set_visible(False)
    for n,v in zip(loads,means):chart.annotate(f'{v:.1f}%',(n,v),xytext=(0,15),textcoords='offset points',ha='center',fontsize=16,color=TEAL,weight='bold')
    chart.legend(loc='lower left',frameon=False,fontsize=16)
    txt(ax,70,795,'20 KiB of fast state per stream',25,INK,'bold')
    txt(ax,70,837,'Memory weights plus momentum only; inputs, shared controller and replay log are additional.',17,MUTED)
    footer(ax,'Training streams contained 16 associations. Results are task-specific; no token savings or serving advantage was measured.')
    save(f,'03-memory-capacity')

if __name__=='__main__':architecture();results();capacity();print('Wrote three PNG/SVG figure pairs.')
