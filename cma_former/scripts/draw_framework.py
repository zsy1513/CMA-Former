"""Original vector schematic; colors encode market identity, never measured values."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle


def render_framework(save):
    fig,ax=plt.subplots(figsize=(7.16,3.03))
    fig.subplots_adjust(left=.006,right=.994,bottom=.008,top=.992)
    ax.set(xlim=(-.5,100.5),ylim=(-1,42))
    ax.axis('off')
    ink='#344454';edge='#8795A1';context='#A75C28'
    market_colors=['#507EA4','#49958E','#B08A50','#A67B91']
    market_fills=['#DFEAF4','#DDEFEA','#F4EACF','#F0E2EA']
    markets=['DE-LU','FR','NL','BE']

    def text(x,y,label,size=7,color=ink,weight='normal',ha='center',va='center'):
        return ax.text(x,y,label,fontsize=size,color=color,weight=weight,ha=ha,va=va,
                       linespacing=1.35,zorder=8)
    def box(x,y,w,h,fill,stroke=edge,rounded=False,lw=.7):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.55',
            facecolor=fill,edgecolor=stroke,lw=lw) if rounded else Rectangle((x,y),w,h,
            facecolor=fill,edgecolor=stroke,lw=lw)
        ax.add_patch(patch)
        return patch
    def arrow(start,end,color=ink,lw=.8,style='-|>',zorder=6):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=7,
                                    lw=lw,color=color,zorder=zorder,shrinkA=1.3,shrinkB=1.3))
    def route(points,color=ink,lw=.8):
        ax.plot(*zip(*points),color=color,lw=lw,zorder=4)
    def add(x,y,r=1.05):
        ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=ink,lw=.9,zorder=7))
        ax.plot([x-r*.54,x+r*.54],[y,y],color=ink,lw=.8,zorder=8)
        ax.plot([x,x],[y-r*.54,y+r*.54],color=ink,lw=.8,zorder=8)

    # Unequal-span conceptual stages, not comparable quantitative plot panels.
    for x,w,fill,title in [(0,35,'#F3F6FA','(a) Market-wise encoding'),
            (38,35,'#F0F7F5','(b) Context-conditioned exchange'),
            (76,24,'#F5F6F2','(c) Residual prediction')]:
        box(x,10,w,28,fill,stroke='#CDD5DA',rounded=True,lw=.65)
        text(x+.5,40.3,title,size=8.2,weight='bold',ha='left')

    # Only a few symbolic tokens are drawn; the label gives the true token count.
    text(17.5,35.7,'336 h history  |  41 patches',size=7.5,weight='bold')
    for m,y in enumerate([28,24,20,16]):
        text(1.3,y,markets[m],size=7,ha='left',color=market_colors[m],weight='bold')
        for k in range(4):
            box(9+k*2.55,y-1.05,2.05,2.1,market_fills[m],market_colors[m],lw=.55)
        text(20.1,y,'...',size=7,color=market_colors[m])
        box(22,y-1.05,2.05,2.1,market_fills[m],market_colors[m],lw=.55)
        arrow((24.3,y),(26.7,y),market_colors[m],lw=.65)
    route([(27.4,30.9),(33.9,30.9),(33.9,14.1)],color='#718599',lw=.7)
    route([(26.9,30.4),(33.4,30.4),(33.4,13.6)],color='#718599',lw=.7)
    box(26.4,13.1,6.5,16.8,'#FFFFFF',stroke='#718599')
    text(29.65,21.6,'Shared\ntemporal\nencoder\n\n3 blocks',size=7)
    text(17.5,11.4,'P = 16, S = 8  |  market + position embeddings',size=7)

    # The addition precedes all query/key/value projections.
    arrow((33.2,22),(38.25,22))
    add(39.5,22)
    arrow((40.7,22),(43.0,22))
    text(47.6,31.2,'Q / K / V',size=7.3,weight='bold')
    text(47,35.3,'Learned\npairwise gate',size=7)
    for i in range(4):
        for j in range(4):
            box(56.1+j*1.05,33.2+i*1.05,.92,.92,'#DCEAE5',stroke='#70958A',lw=.35)
    arrow((60.5,35.2),(64.4,35.2),color='#5E877E',lw=.7)
    arrow((64.4,35.2),(64.4,30.2),color='#5E877E',lw=.7)
    ys=[28,24,20,16]
    for y1 in ys:
        for y2 in ys:
            route([(46.1,y1),(66.4,y2)],color='#AEC3C0',lw=.42)
    for m,y in enumerate(ys):
        # Color follows identity across the shared temporal and market axes.
        ax.add_patch(Circle((45.3,y),.72,facecolor=market_fills[m],edgecolor=market_colors[m],lw=.7,zorder=7))
        ax.add_patch(Circle((67.2,y),.72,facecolor=market_fills[m],edgecolor=market_colors[m],lw=.7,zorder=7))
    text(55.5,12.2,'4 markets / patch  |  8 heads / 2 blocks',size=7)
    route([(68.6,22),(74.1,22),(74.1,30.6)],lw=.8)
    arrow((74.1,30.6),(78,30.6))

    box(78,27,20,7.2,'#E7ECE1',stroke='#8D9C80',rounded=True)
    text(88,31.5,'Last-token head',size=7.8,weight='bold')
    text(88,28.9,'24 x 3 residuals',size=7)
    arrow((88,26.8),(88,23.2))
    add(88,22,r=1.05)
    arrow((88,20.8),(88,18.5))
    text(88,16.8,'24-hour price forecasts',size=7.4,weight='bold')
    for k,(label,fill) in enumerate([('0.1','#E8EFED'),('0.5','#CBDFD9'),('0.9','#E8EFED')]):
        box(78.5+6.3*k,11.8,5.75,2.5,fill,stroke='#76968D',lw=.5)
        text(81.35+6.3*k,13.05,label,size=7)

    # Known covariates feed both early context and the common output adapter.
    box(0,1,32,6.6,'#FBF2E5',stroke='#C6A37E',rounded=True)
    text(16,5.6,'Known delivery-day context',size=7.5,weight='bold',color=context)
    text(16,2.8,'D-1 price, load / RES forecasts, calendar',size=7)
    arrow((32,4.2),(38,4.2),context)
    box(38,1,25,6.6,'#FBF2E5',stroke='#C6A37E',rounded=True)
    text(50.5,5.5,'24-hour context projection',size=7.4,weight='bold',color=context)
    text(50.5,2.7,'Broadcast over historical patches',size=7)
    route([(50.5,7.8),(50.5,9),(39.5,9),(39.5,19.3)],context)
    arrow((39.5,19.3),(39.5,20.8),context)
    route([(35.2,4.2),(35.2,-.2),(73.5,-.2),(73.5,4.2)],context)
    arrow((73.5,4.2),(76,4.2),context)
    box(76,1,24,6.6,'#FBF2E5',stroke='#C6A37E',rounded=True)
    text(88,5.5,'Common output adapter',size=7.3,weight='bold',color=context)
    text(88,2.7,'Hourly correction + D-1 curve',size=7)
    route([(100,4.2),(100.35,4.2),(100.35,22),(90.4,22)],context)
    arrow((90.4,22),(89.2,22),context)
    save(fig,'fig1_architecture')
