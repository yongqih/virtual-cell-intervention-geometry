"""Assemble S2 and S15 from the manuscript's archived scores and plotting code."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from settings import INPUT, OUTPUT as OUT, CHECKS as LAYOUT
def render_s2():
    import quality as source
    source.OUT=OUT
    source.LAYOUT=LAYOUT
    table=pd.read_csv(INPUT/"fig12/S2_source_data.csv")
    def read(name):
        return table if name.endswith("reliability.csv") else pd.read_csv(INPUT/"fig12"/name)
    source.read_table=read
    def bars(ax, frame, metric, scale, title, letter, limit):
        order=["experimental_split_half",*source.MODELS]
        rows=frame.query("scope=='two_fold_mean' and min_cells==20 and metric==@metric and scale==@scale").set_index("entity").loc[order]
        y=np.arange(len(rows))
        v=rows.estimate
        ax.barh(y,v,color=[source.REF]+[source.COLORS[m] for m in source.MODELS],height=.61,zorder=2)
        ax.errorbar(v,y,xerr=[v-rows.ci_low,rows.ci_high-v],fmt="none",ecolor=source.INK,elinewidth=.75,capsize=1.8,capthick=.65,zorder=4)
        if scale=="raw":
            chance=rows.chance_raw.iloc[0]
            ax.axvline(chance,color="#89949C",ls=(0,(3,2)),lw=.8,zorder=1)
            ax.text(chance+.014,7.79,"Chance",fontsize=5.8,color=source.MUTED,va="center")
        ax.set(yticks=y,yticklabels=["Split-half",*source.MODELS],ylim=(8.2,-.7),xlim=(0,limit),xlabel="Score",
               xticks=[0,.25,.5,.75,1] if limit>1 else [0,.2,.4,.6])
        source.label(ax,letter,title)
        source.style(ax)
    source.reliability_bars=bars
    source.supplementary2()


def render_s15():
    import quality  # Common supplementary axis styling.
    import fig6 as source
    source.OUT=OUT
    source.LAYOUT=LAYOUT
    class AlignedPage(source.Page):
        def panel(self,letter,title,x,y,center=None):
            axis=self.axes[-1].get_position()
            # Anchor each letter to its own axis, with the metric centered above it.
            super().panel(letter,title,axis.x0*self.w-4.8,y,(axis.x0+axis.x1)*self.w/2)
        def text(self,x,y,s,*args,**kwargs):
            if s=="Full-input residual − Copy + mean":
                s="Full-input residual versus mean-corrected Copy"
            return super().text(x,y,s,*args,**kwargs)
    source.old.Page=AlignedPage
    source.old.supplement5()
