"""Original editable biological process cartoons, not molecular reconstructions.

Protein locations and
outlines are illustrative and do not assert subunit geometry or stoichiometry.
"""
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Ellipse, Circle, FancyBboxPatch


class BiologyArt:
    def __init__(self,ax,ink,blue,orange):
        self.ax=ax; self.ink=ink; self.blue=blue; self.orange=orange
        self.edge='#8C969C';self.pale='#F1F3F4';self.fill='#DFE4E7'

    def path(self,cx,cy,start,segments,fill='none',edge=None,lw=.65,scale=1,z=2):
        pts=[start];codes=[Path.MOVETO]
        for seg in segments:
            pts.extend(seg);codes.extend([Path.CURVE4]*3)
        if fill!='none':
            pts.append(start);codes.append(Path.CLOSEPOLY)
        pts=np.array(pts)*scale+np.array([cx,cy])
        self.ax.add_patch(PathPatch(Path(pts,codes),fc=fill,ec=edge or self.edge,
                                   lw=lw,joinstyle='round',capstyle='round',zorder=z))

    def ellipse(self,cx,cy,w,h,fill,edge=None,lw=.55,angle=0,z=3):
        self.ax.add_patch(Ellipse((cx,cy),w,h,angle=angle,fc=fill,ec=edge or self.edge,lw=lw,zorder=z))

    def splice(self,cx,cy):
        # Ribonucleoprotein envelope around a folded RNA substrate.
        self.path(cx,cy,(-7,-.5),[
            [(-8.5,1),(-7.4,3.3),(-5.7,3.6)],
            [(-5.3,5),(-2.8,5.8),(-1.4,4.9)],
            [(.7,6.2),(3.5,4.7),(3.9,3.7)],
            [(6.5,4.1),(8.2,2.1),(7.2,.5)],
            [(8.1,-1.8),(5.5,-3.4),(3.7,-2.5)],
            [(1.3,-4.6),(-1.4,-4.5),(-2.8,-2.8)],
            [(-5.5,-3.7),(-8.1,-2.6),(-7,-.5)]],self.pale,self.edge,.6)
        # Bound RNA loops through the spliceosome and forms an intron lariat.
        self.path(cx,cy,(-11,-3.7),[
            [(-7,-3.7),(-5,-3.3),(-3.5,-1.7)],
            [(-2,.4),(-4.2,3.4),(-.8,3.7)],
            [(3.3,4),(4.6,.5),(2.1,-.6)],
            [(-.1,-1.6),(-2.4,.2),(-.8,1.7)],
            [(1,3.1),(2.5,1.6),(1.4,.5)],
            [(.5,-.7),(3.9,-3.7),(11,-3.7)]],edge='#69767D',lw=.9,z=4)
        # Distinct dark and blue protein regions; positions are schematic.
        self.path(cx,cy,(-5.9,1.8),[
            [(-7.2,2.3),(-6.6,4.6),(-4.5,4.5)],
            [(-3.3,5.2),(-2.0,3.8),(-2.5,2.7)],
            [(-2.4,1.2),(-4.7,.4),(-5.9,1.8)]],self.ink,self.ink,.5,z=5)
        self.path(cx,cy,(3.7,-.7),[
            [(3.1,.6),(4.5,2.4),(6.1,1.9)],
            [(7.4,1.4),(7.5,-.4),(6.3,-1.2)],
            [(5.7,-2.2),(4.1,-2.3),(3.7,-.7)]],'#B9CDD8',self.blue,.65,z=5)
        self.path(cx,cy,(-5,-1.1),[[(-3.7,-.6),(-3.8,-2.2),(-2.7,-2.5)]],edge='#B2BBC0',lw=.5,z=5)
        for xx in [cx-10,cx+6.5]:
            self.ax.add_patch(FancyBboxPatch((xx,cy-4.35),3.5,1.3,
                boxstyle='round,pad=0,rounding_size=.24',fc=self.fill,ec=self.edge,lw=.55,zorder=5))

    def initiation(self,cx,cy):
        # 40S body with head and mRNA cleft; an eIF3-associated region is ochre.
        self.path(cx,cy,(-7.9,-1.5),[
            [(-9,1.8),(-7.5,3.7),(-4.5,3.4)],
            [(-3.6,5.9),(.5,5.9),(2.2,3.6)],
            [(5.6,4.4),(8,2.1),(7.8,-.4)],
            [(8.2,-3),(4.1,-4.5),(1.4,-3.9)],
            [(-.9,-5.4),(-5.7,-4.8),(-7.9,-1.5)]],self.pale,self.edge,.65)
        self.path(cx,cy,(-4.6,3.3),[[(-3.5,2.2),(-3.1,.8),(-3.8,-.7)]],edge='#BBC3C8',lw=.5)
        self.path(cx,cy,(2.1,3.3),[[(2.8,1.5),(1.6,-.5),(2.8,-2.2)]],edge='#BBC3C8',lw=.5)
        self.path(cx,cy,(-11,-.9),[[(-6,-2.2),(-1,-1.5),(2,-.7)],[(5,.1),(7,.2),(11,-.7)]],
                  edge=self.ink,lw=.8,z=5)
        # Folded initiator tRNA at the decoding cleft, kept neutral.
        self.path(cx,cy,(-.9,-.9),[
            [(-2.6,.3),(-1.8,1.4),(-.5,1.2)],
            [(-1.3,2.6),(.2,3.1),(.7,1.9)],
            [(2.5,2.5),(3,1.2),(1.3,.7)],
            [(1.2,-.1),(.1,-.8),(-.9,-.9)]],edge='#708088',lw=.75,z=6)
        self.path(cx,cy,(5.5,1.8),[
            [(6.7,3.4),(8.2,3),(8.5,1.9)],
            [(10,2),(10.5,.2),(9.4,-.6)],
            [(10.1,-2),(8.4,-3),(7.3,-2.3)],
            [(5.4,-2.9),(4.9,-1.2),(5.7,-.6)],
            [(4.7,.3),(4.6,1.5),(5.5,1.8)]],'#F7E7D8',self.orange,.7,z=6)
        self.ellipse(cx+8.1,cy+.1,1.5,1.8,self.orange,self.orange,.4,z=7)
        for x in [-9,-7,6,8]:
            self.ax.plot([cx+x,cx+x],[cy-1.6,cy-2.25],color='#8C969C',lw=.4,zorder=6)

    def mitochondrion(self,cx,cy):
        # Bean-shaped outer membrane and a continuous folded inner membrane.
        self.path(cx,cy,(-12,-.5),[
            [(-12.4,3.7),(-8.4,5.7),(-3.6,5.5)],
            [(1.8,5.6),(7.7,3.5),(10.4,1.8)],
            [(14,-.5),(11.7,-4.2),(7.1,-5)],
            [(1.2,-6.2),(-6.7,-5.2),(-10.3,-3.1)],
            [(-11.6,-2.5),(-12.1,-1.4),(-12,-.5)]],'#EBEEF0','#707D85',.75)
        self.path(cx,cy,(-10.3,-.7),[
            [(-10.8,1.8),(-9,3.4),(-7,3.7)],
            [(-6.2,3.8),(-5.7,3.9),(-5.1,3.9)],
            [(-4.8,2),(-6,-1.2),(-4.9,-1.9)],
            [(-3.4,-1.4),(-3.7,2.8),(-3.5,3.9)],
            [(-2.7,4),(-1.7,3.9),(-1,3.7)],
            [(-.8,2),(-2,-2.3),(-.6,-2.7)],
            [(.9,-2.2),(.6,1.7),(.8,3.5)],
            [(1.8,3.3),(2.6,3.1),(3.4,2.8)],
            [(3.6,1),(2.3,-2.9),(3.7,-3.1)],
            [(5,-2.8),(4.9,1.1),(5,2.3)],
            [(7,1.5),(8.5,1.4),(9.6,.6)],
            [(10,-1),(8.8,-2.9),(6.8,-3.8)],
            [(1,-4.8),(-5,-4),(-8.8,-2.6)],
            [(-10,-2),(-10.4,-1.4),(-10.3,-.7)]],'#FAFBFB','#8B989F',.65)
        # A membrane-associated respiratory complex, with identity accents.
        self.path(cx,cy,(-.7,1.2),[
            [(-1.7,1.8),(-3,1.1),(-2.9,.1)],
            [(-3.2,-1),(-1.8,-1.8),(-.9,-1.2)],
            [(.1,-1.8),(1.5,-.9),(1.3,.1)],
            [(1.4,1.2),(.2,1.8),(-.7,1.2)]],'#C1CBD0','#6E7F89',.5,z=5)
        self.ellipse(cx-1.75,cy+.55,1.2,2.1,self.ink,self.ink,.4,angle=-12,z=6)
        self.ellipse(cx+.25,cy+.45,1.35,2.3,self.blue,self.blue,.4,angle=10,z=6)

    def chromatin(self,cx,cy):
        # DNA wraps histone cores; a multi-domain remodeler embraces the centre.
        for x,y in [(-8,-1.5),(0,-2.2),(8,-1.4)]:
            self.ellipse(cx+x,cy+y,4.5,3.7,'#DEE3E6','#96A0A6',.55)
            self.ellipse(cx+x-.2,cy+y+.7,3.6,1.5,'#EEF1F2','#AAB2B7',.4)
        self.path(cx,cy,(-12,-2.1),[
            [(-10.2,-2.1),(-10.7,1.2),(-8,1.1)],
            [(-4.9,1),(-4.8,-4.6),(-8,-4.3)],
            [(-11,-4),(-10,-1),(-5.7,-1.1)],
            [(-3.9,-1.1),(-2.8,.6),(0,.5)],
            [(3,.3),(3,-5.3),(0,-5)],
            [(-3.1,-4.8),(-2.8,-1.6),(3.2,-1.5)],
            [(5,-1.4),(5.1,1.3),(8,1.2)],
            [(11.1,1),(11.2,-4.5),(8,-4.1)],
            [(5,-3.8),(5.6,-.9),(12,-.6)]],edge='#6C7E89',lw=.85,z=5)
        self.path(cx,cy,(-3.9,1),[
            [(-6.3,2.3),(-4.9,4.5),(-2.8,4.6)],
            [(-1.9,6.2),(.9,6.2),(2.2,4.7)],
            [(4.7,5.2),(6.1,3.2),(4.8,1.7)],
            [(4.5,-.5),(2.7,-1),(1.9,.4)],
            [(.6,1.2),(-1.5,1.5),(-2,.3)],
            [(-3.2,-.7),(-4.9,-.2),(-3.9,1)]],'#F7EBDD',self.orange,.7,z=6)
        self.path(cx,cy,(-1.8,4.3),[[(-.2,3.8),(-.6,2.1),(.2,1.4)]],edge='#D5A882',lw=.55,z=7)
        self.path(cx,cy,(2.3,4.4),[[(1.6,3.2),(2.7,2),(3.8,2.1)]],edge='#D5A882',lw=.55,z=7)
        self.ellipse(cx-3.6,cy+2.8,2.1,1.7,self.orange,self.orange,.4,angle=18,z=7)
