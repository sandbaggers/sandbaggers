"""Plot measured training-dose curves: easy and hard recall of two organisms per step, on one axis. The callers
(pirate_rv5.py, targeted_rv5.py) supply the data, doses and models."""
from pathlib import Path
import os,sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/elkmo-matplotlib')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from plots import style

def draw(ax,data,tier,doses,models):
 """data[label][dose][tier] = {mean, lo, hi} (fractions); models [(label, arm, colour, marker)]. Easy dashed, hollow."""
 x=list(range(len(doses)))
 for label,_,colour,marker in models:
  values=[data[label][d][tier] for d in doses]
  ax.fill_between(x,[100*v['lo'] for v in values],[100*v['hi'] for v in values],color=colour,alpha=.08 if tier=='easy' else .12,linewidth=0)
  ax.plot(x,[100*v['mean'] for v in values],label=f'{label} · {tier.capitalize()}',color=colour,marker=marker,markersize=10,markerfacecolor='white' if tier=='easy' else colour,markeredgecolor=colour if tier=='easy' else 'white',markeredgewidth=1.4,linewidth=2.2,linestyle='--' if tier=='easy' else '-')
 ax.set(xlim=(-.25,len(doses)-.75),ylim=(104,-2.5),xticks=x,xticklabels=doses,yticks=[0,20,40,60,80,100],xlabel='Training steps',ylabel='Vulnerability recall (%)')
 ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.grid(axis='x',visible=False);ax.spines['left'].set_bounds(0,100)

def plot(data,doses,models,name,out):
 """Both tiers of every model on one axis -> <out>/{png,pdf}/<name>.*"""
 style.apply_style()
 plt.rcParams['figure.constrained_layout.use']=False
 plt.rcParams['figure.autolayout']=False
 fig,ax=plt.subplots(figsize=(16,7.5),layout='none')
 for tier in ['hard','easy']:draw(ax,data,tier,doses,models)
 fig.subplots_adjust(left=.095,right=.985,bottom=.16,top=.76)
 handles,labels=ax.get_legend_handles_labels();order=[0,2,1,3]
 leg=fig.legend([handles[i] for i in order],[labels[i] for i in order],ncol=4,loc='upper center',bbox_to_anchor=(.5,.99),handlelength=2.1,columnspacing=1.2,fontsize=20,frameon=True)
 fig.canvas.draw()
 assert leg.get_window_extent(fig.canvas.get_renderer()).y0 > ax.get_window_extent().y1 + 20, 'Legend must clear plot'
 style.save(fig,name,out,formats=('png','pdf'));plt.close(fig)
 print(Path(out)/f'png/{name}.png')
