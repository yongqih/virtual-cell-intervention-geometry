"""Redraw the temporal schematic and reuse the quantitative panels from Fig. 5."""
from pathlib import Path
import importlib.util
import math

import matplotlib as mpl
import numpy as np
from matplotlib.path import Path as VectorPath
from matplotlib.patches import PathPatch, Polygon, Circle, FancyArrowPatch
from matplotlib.transforms import Bbox


from settings import INPUT, OUTPUT as OUT, CHECKS as LAYOUT
import temporal as original
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 6.5, "pdf.fonttype": 42, "svg.fonttype": "none",
})
INK = "#343E42"
BLUE = "#628A9D"
SLATE = "#50656D"
GOLD = "#AE805B"
GLYPHS = ["o", "s", "^"]


def curve(ax, points, **style):
    path = VectorPath(points, [VectorPath.MOVETO] + [VectorPath.CURVE4] * 3)
    ax.add_patch(PathPatch(path, fill=False, capstyle="round", **style))


def label(ax, x, y, value, size=6.4, **style):
    ax.text(x, y, value, fontsize=size, ha="center", va="center",
            color=style.pop("color", INK), **style)


def time_slice(ax, x):
    """Open coordinate frame for a schematic projection of response space."""
    ax.plot([x-10,x-13.5,x+11.8],[132.0,116.4,113.6],color="#C5D1D1",
            lw=.65,solid_capstyle="round",zorder=1)


def glyph(ax, xy, shape, color, filled, zorder=6):
    ax.plot(*xy, marker=shape, linestyle="none", ms=3.8,
            markeredgecolor=color, markeredgewidth=.82,
            markerfacecolor=color if filled else "white", zorder=zorder)


def constellation(ax, points, color, observed):
    closed = np.vstack([points, points[:1]])
    ax.add_patch(Polygon(points,facecolor=color,alpha=.075,edgecolor="none",zorder=3))
    ax.plot(*closed.T, color=color, linewidth=.80 if observed else 1.0,
            solid_joinstyle="round", linestyle="-" if observed else (0, (2.4, 1.7)),
            zorder=4 if observed else 5)
    for point, shape in zip(points, GLYPHS):
        glyph(ax, point, shape, color, observed, 6 if observed else 7)


def bio_icon(ax, x, y, measured):
    """Small monoline illustrations of a treated cell and a chemical annotation."""
    if measured:
        angles = np.linspace(0, 2*math.pi, 120)
        radius = 1 + .07*np.cos(3*angles-.9) + .035*np.sin(2*angles)
        ax.fill(x+3.8*radius*np.cos(angles), y+2.7*radius*np.sin(angles),
                facecolor="#F0F5F3", edgecolor="#7D9D91", linewidth=.65)
        ax.add_patch(Circle((x-.55, y+.15), 1.12, facecolor="#D5E4DE",
                            edgecolor="#79978C", linewidth=.5))
        for dx, dy in [(1.1,1.55),(2,.1),(1.1,-1.4),(-2,-.6)]:
            ax.add_patch(Circle((x+dx,y+dy),.2,facecolor="#7D9D91",edgecolor="none"))
    else:
        angle = np.arange(7)*math.pi/3 + math.pi/6
        hexagon = np.column_stack([x+1.65*np.cos(angle), y+1.65*np.sin(angle)])
        ax.plot(*hexagon.T, color=GOLD, lw=.75)
        ax.plot([x+1.43,x+2.66,x+3.60],[y+.83,y+1.55,y+.97],color=GOLD,lw=.75)
        ax.plot([x-1.42,x-2.8],[y-.83,y-1.54],color=GOLD,lw=.75)
        ax.plot([x-1.01,x-1.01],[y-.6,y+.60],color=GOLD,lw=.45)


def temporal_scene(ax, x0, heading, start, observed, measured):
    start_x, end_x = x0+19, x0+66
    time_slice(ax, start_x)
    time_slice(ax, end_x)
    route_color = BLUE if measured else GOLD
    bio_icon(ax, x0+(25.5 if measured else 22), 139.5, measured)
    label(ax, x0+46, 139.5, heading, 6.8, fontweight="bold")
    start_pts = start + [start_x, 123]
    # Schematic coordinates only. Both starting states pass through the same map.
    temporal_map = np.array([[.94, -.22], [.13, .97]])
    finish = start @ temporal_map.T + [end_x, 123]
    target = observed + [end_x, 123]
    for idx,(src,dst) in enumerate(zip(start_pts, finish)):
        # Soft association lines carry drug identity between the two time slices.
        curve(ax, [src, src+[18, 3.4-idx*.45], dst-[16, 2.9], dst],
              edgecolor=route_color, linewidth=.65, alpha=.44, zorder=2)
    constellation(ax, start_pts, route_color, observed=measured)
    constellation(ax, target, INK, observed=True)
    constellation(ax, finish, route_color, observed=False)
    label(ax, start_x, 109.2, "24 h", 6.3)
    label(ax, end_x, 109.2, "48 h", 6.3)
    ax.add_patch(FancyArrowPatch((start_x+6.4,109.2),(end_x-6.6,109.2),
                                 arrowstyle="-|>",mutation_scale=5,
                                 color="#A4B3B5",linewidth=.54))


def panel_a(sheet):
    ax = sheet.cv
    sheet.title("a", "Predicting drug responses with the same temporal model", 3,150,177)
    # Each shape is one illustrative drug. The observed late positions are shared.
    actual = np.array([[-7.0,1.6],[-5.3,-3.8],[7.2,3.6]])
    measured = np.array([[-7.4,3.5],[-5.2,-1.8],[6.8,4.5]])
    estimated = np.array([[-6.0,4.0],[5.5,-1.2],[2.2,-3.5]])
    temporal_scene(ax, 4, "24 h state from drug annotations", estimated, actual, False)
    temporal_scene(ax, 93, "Measured 24 h proteome", measured, actual, True)
    for a,b,ls,title in [(50,56,"-","Observed"),(84,90,(0,(2.4,1.7)),"Predicted")]:
        ax.plot([a,b],[101.4,101.4],color=INK,lw=.9,linestyle=ls)
        sheet.t(b+2.4,101.4,title,size=5.8,ha="left")
    for i,shape in enumerate(GLYPHS):
        glyph(ax,(116.7+i*2.2,101.4),shape,SLATE,True)
    sheet.t(124.5,101.4,"Different drugs",size=5.8,ha="left")


def export_figure():
    sheet = original.Sheet(155)
    panel_a(sheet)
    crop = Bbox.from_extents(0, 97.5/25.4, 183/25.4, 155/25.4)
    sheet.fig.savefig(OUT/"Figure_5a.png",bbox_inches=crop,pad_inches=0,dpi=600)
    sheet.fig.savefig(OUT/"Figure_5a.svg",bbox_inches=crop,pad_inches=0)
    sheet.fig.savefig(OUT/"Figure_5a.pdf",bbox_inches=crop,pad_inches=0)

    # Existing quantitative panel calls reproduce the approved layout and tables.
    sheet.title('b','Measured-entry gain',3,94,84)
    a=sheet.ax(24,57,60,29,'b')
    for y,m,c in [(1,'ode',original.BLUE),(0,'ridge',original.OCHRE)]:original.point(a,original.pair(m+'_measured',m+'_predicted'),y,c,True)
    original.clean(a,'Δρ',[1,0],['ODE','Ridge'],(-.08,.53));a.set_ylim(-.65,1.65)
    sheet.title('c','Matched temporal baselines',94,94,86)
    a=sheet.ax(128,57,49,29,'c')
    ms=['direct_metadata_late','persistence','ode_measured','ridge_measured','full_ridge']
    labels=['Metadata','Persistence','ODE','Coordinate Ridge','Full Ridge']
    for y,m,c in zip(range(4,-1,-1),ms,[original.GRAY,original.TEAL,original.BLUE,original.OCHRE,original.INK]):original.point(a,original.geo(m),y,c)
    original.clean(a,'48 h global ρ',range(4,-1,-1),labels,(0,.65));a.set_ylim(-.5,4.5)
    sheet.title('d','Context-specific recovery',3,42,177)
    cells=['HCC1143','HCC1806','MDA-MB-453-1']
    for x,m,c in [(32,'ode',original.BLUE),(121,'ridge',original.OCHRE)]:
     a=sheet.ax(x,11,56,22,'d_'+m)
     for y,cell in zip([2,1,0],cells):original.point(a,original.pair(m+'_measured',m+'_predicted',cell),y,c,True)
     original.clean(a,'Δρ',[2,1,0],cells,(-.6,.9));a.set_ylim(-.5,2.5);a.set_xticks([-.5,0,.5])
     a.set_title('ODE' if m=='ode' else 'Ridge',pad=4)
    sheet.save('Figure_5',[['b','c'],['d_ode','d_ridge']])
