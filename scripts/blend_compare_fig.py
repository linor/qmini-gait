"""Draw the parameter-blend vs pose-blend comparison."""
import os, sys, json, base64
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.gridspec as gs

HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import blend_compare as BC, gait_core as GC, gait_planner as GP

B=json.load(open(os.path.join(HERE,"qmini_mesh.json")))
P0=np.frombuffer(base64.b64decode(B["pos"]),dtype=np.float32).reshape(-1,3).astype(float)
IDX=np.frombuffer(base64.b64decode(B["idx"]),dtype=np.uint32).reshape(-1,3).astype(int)
PARTS=B["parts"]
JN=["Root"]+[f"{s}_{j}" for s in "LR" for j in GC.CHAIN]
R2V=lambda p: np.stack([-p[...,1],p[...,0],p[...,2]],-1)
C2R=np.array([[0.,1,0],[-1,0,0],[0,0,1]])            # char -> robot

def world_frames(q,euler,ground):
    """joint world transforms in ROBOT cm, root placed so the support foot is
    on z=0 and the body is centred laterally"""
    R0=GC.rot_zyx(0.0,euler[1],euler[2])
    P={"Root":np.zeros(3)},
    out={}
    out["Root"]=(R0,np.array([0.,0.,-ground]))
    for s in "LR":
        leg=GP.LEGS[s]; R=R0; p=np.array([0.,0.,-ground])+R0@leg.a["hip_yaw"]
        out[f"{s}_hip_yaw"]=(R@GC.rot_axis(leg.ax["hip_yaw"],q[s][0]),p)
        R=R@GC.rot_axis(leg.ax["hip_yaw"],q[s][0])
        for i,j in enumerate(GC.CHAIN[1:],start=1):
            p=p+R@leg.local[j]; R=R@GC.rot_axis(leg.ax[j],q[s][i])
            out[f"{s}_{j}"]=(R,p)
    # convert char (m) -> robot (cm)
    return {k:(C2R@R@C2R.T, C2R@p*100) for k,(R,p) in out.items()}

def render(ax,q,euler,ground,title,az=-1.15,el=0.16,d=118,W=520,H=430,hi=None,
           target=None,gridstep=14):
    F=world_frames(q,euler,ground)
    Pw=np.zeros_like(P0); fc=[]
    for p in PARTS:
        R,t=F[p["joint"]]; sl=slice(p["v0"],p["v0"]+p["nv"])
        Pw[sl]=(P0[sl]@R.T)+t; fc+=[p["joint"]]*p["nf"]
    Pv=R2V(Pw)
    ce,se=np.cos(el),np.sin(el)
    center=np.array([0.,0.,17.]) if target is None else np.asarray(target,float)
    eye=center+np.array([d*ce*np.cos(az),d*ce*np.sin(az),d*se])
    fw=center-eye; fw/=np.linalg.norm(fw)
    r=np.cross(fw,[0,0,1.]); r/=np.linalg.norm(r); u=np.cross(r,fw)
    fl=min(W,H)*1.45
    def PR(p):
        dd=p-eye; z=dd@fw
        return np.stack([W/2+fl*(dd@r)/z,H/2-fl*(dd@u)/z],-1),z
    # ground plane grid
    gsz=gridstep
    for k in range(-9,10):
        for sg in ([[center[0]-140,center[1]+k*gsz,0],[center[0]+140,center[1]+k*gsz,0]],
                   [[center[0]+k*gsz,center[1]-140,0],[center[0]+k*gsz,center[1]+140,0]]):
            s2,_=PR(np.array(sg,float)); ax.plot(s2[:,0],s2[:,1],color="#242a38",lw=.7,zorder=1)
    s2,_=PR(np.array([[center[0]-140,center[1],0],[center[0]+140,center[1],0]],float))
    ax.plot(s2[:,0],s2[:,1],color="#4a5570",lw=1.4,zorder=2)
    S,Z=PR(Pv); tri=S[IDX]; zc=Z[IDX].mean(1)
    n=np.cross(Pv[IDX][:,1]-Pv[IDX][:,0],Pv[IDX][:,2]-Pv[IDX][:,0])
    n/=(np.linalg.norm(n,axis=1,keepdims=True)+1e-9)
    L=np.array([.35,.55,.90]); L/=np.linalg.norm(L); lam=np.clip(n@L,0,1)
    cols=[]
    for i,j in enumerate(fc):
        if j=="Root": c=np.array([.62,.65,.72])
        else:
            c=np.array([.82,.45,.42]) if j[0]=="L" else np.array([.40,.58,.85])
            if not("ankle" in j or "knee" in j): c=c*.82+np.array([.62,.65,.72])*.18
        if hi and j==hi: c=np.array([.95,.64,.34])
        cols.append(np.clip(c*(.30+.16*n[i][2]+.72*lam[i]),0,1))
    o=np.argsort(-zc)
    ax.add_collection(PolyCollection(tri[o],facecolors=np.array(cols)[o],
                                     edgecolors='none',zorder=3))
    ax.set_xlim(0,W); ax.set_ylim(H,0); ax.set_aspect("equal")
    ax.axis("off"); ax.set_facecolor("#12141a")
    ax.set_title(title,color="#c8cfdd",fontsize=10.5,pad=6)
    return PR

PAIRS=[("forward_slow","forward_fast","adjacent ẋ pair"),
       ("forward_fast","left_strafe","forward × strafe"),
       ("forward_fast","left_turn","forward × turn"),
       ("reverse_fast","left_turn","reverse × turn")]

print("running blends ...", flush=True)
CASES=[]
for a,b,lab in PAIRS:
    c=BC.run_case(a,b,0.5,lab); c["short"]=lab; CASES.append(c)
    pc=np.array([r[1]['clearance'] for r in c['rows']])*1000
    bc=np.array([r[2]['clearance'] for r in c['rows']])*1000
    print(f"  {lab:22s} param {pc.min():+6.2f} mm   pose {bc.min():+6.2f} mm",flush=True)

worst=CASES[-1]
phis=np.array([r[0] for r in worst["rows"]])
bcl=np.array([r[2]["clearance"] for r in worst["rows"]])
k=int(np.argmin(bcl))
phi_w=phis[k]; mp=worst["rows"][k][1]; mb=worst["rows"][k][2]
print(f"worst phase phi={phi_w:.3f}  state={mb['state']}  swing={mb['swing']}")

fig=plt.figure(figsize=(17.5,9.2),facecolor="#12141a")
G=gs.GridSpec(2,3,height_ratios=[1.05,1],hspace=.24,wspace=.13,
              left=.045,right=.985,top=.895,bottom=.095)

axA=fig.add_subplot(G[0,0]); axB=fig.add_subplot(G[0,1])
sw=mb["swing"]

def cam_for(target,az,el,d,W,H,fl_scale=1.45):
    center=np.asarray(target,float)
    ce,se=np.cos(el),np.sin(el)
    eye=center+np.array([d*ce*np.cos(az),d*ce*np.sin(az),d*se])
    fw=center-eye; fw/=np.linalg.norm(fw)
    r=np.cross(fw,[0,0,1.]); r/=np.linalg.norm(r); u=np.cross(r,fw)
    fl=min(W,H)*fl_scale
    def PR(p):
        p=np.atleast_2d(np.asarray(p,float)); dd=p-eye; z=dd@fw
        return np.stack([W/2+fl*(dd@r)/z,H/2-fl*(dd@u)/z],-1),z
    return PR

def part_tris(m,euler,ground,joints):
    """world-space triangles (view frame) of the selected joints' parts"""
    F=world_frames(m["q"],euler,ground)
    out=[]
    for p in PARTS:
        if p["joint"] not in joints: continue
        R,t=F[p["joint"]]; sl=slice(p["v0"],p["v0"]+p["nv"])
        V=R2V((P0[sl]@R.T)+t)
        loc=IDX[(IDX[:,0]>=p["v0"])&(IDX[:,0]<p["v0"]+p["nv"])]-p["v0"]
        out.append((V,loc))
    return out

def shade(ax,PR,chunks,base,alpha,zorder):
    tris=[];cols=[];zs=[]
    for V,F in chunks:
        S,Z=PR(V); t=S[F]; zc=Z[F].mean(1)
        n=np.cross(V[F][:,1]-V[F][:,0],V[F][:,2]-V[F][:,0])
        n/=(np.linalg.norm(n,axis=1,keepdims=True)+1e-9)
        L=np.array([.35,.55,.90]); L/=np.linalg.norm(L); lam=np.clip(n@L,0,1)
        for i in range(len(t)):
            tris.append(t[i]); zs.append(zc[i])
            cols.append(np.clip(np.array(base)*(.34+.14*n[i][2]+.70*lam[i]),0,1))
    o=np.argsort(-np.array(zs))
    ax.add_collection(PolyCollection(np.array(tris)[o],facecolors=np.array(cols)[o],
        edgecolors='none',alpha=alpha,zorder=zorder))

# ---------- context: whole robot ----------
PRfull=render(axA,mp["q"],mp["euler"],mp["ground"],
   "the blended gait, just before right touchdown  (φ = %.2f)" % phi_w,
   d=118,el=0.16,az=-1.15,gridstep=14)
foot_c=(C2R@mb["sole"][sw])*100
tgtv=np.array([float(-foot_c[1]),float(foot_c[0]),3.0])
bx,_=PRfull(np.array([[tgtv[0]-11,tgtv[1]-11,-2],[tgtv[0]+11,tgtv[1]-11,-2],
                      [tgtv[0]+11,tgtv[1]+11,16],[tgtv[0]-11,tgtv[1]+11,16]]))
axA.add_patch(plt.Rectangle((bx[:,0].min(),bx[:,1].min()),
    np.ptp(bx[:,0]),np.ptp(bx[:,1]),fill=False,ec="#e6ebf2",lw=1.2,ls=(0,(4,3)),zorder=8))
axA.text(bx[:,0].mean(),bx[:,1].min()-8,"detail →",color="#e6ebf2",fontsize=9,ha="center")

# ---------- detail: the two swing feet side by side on a shared floor ----------
W2,H2=600,430
AZ,EL,D=-1.30,0.075,48
def cam_axes(center,az,el,d):
    ce,se=np.cos(el),np.sin(el)
    eye=np.asarray(center,float)+np.array([d*ce*np.cos(az),d*ce*np.sin(az),d*se])
    fw=np.asarray(center,float)-eye; fw/=np.linalg.norm(fw)
    r=np.cross(fw,[0,0,1.]); r/=np.linalg.norm(r); u=np.cross(r,fw)
    return eye,fw,r,u
ctr=np.array([tgtv[0],tgtv[1],2.6])
eye,fwv,rv,uv=cam_axes(ctr,AZ,EL,D)
PR2=cam_for(ctr,AZ,EL,D,W2,H2,1.45)
axB.set_xlim(0,W2); axB.set_ylim(H2,0); axB.set_aspect("equal")
axB.axis("off"); axB.set_facecolor("#12141a")

# floor: a line through the feet, laid along the camera's right vector, so it
# projects perfectly horizontal and reads as "the floor"
gl=np.array([ctr+rv*(-46)-np.array([0,0,ctr[2]]), ctr+rv*46-np.array([0,0,ctr[2]])])
sg,_=PR2(gl)
yfloor=float(sg[:,1].mean())
axB.fill_between([0,W2],[yfloor,yfloor],[H2,H2],color="#e0655f",alpha=.11,zorder=0)
axB.plot([0,W2],[yfloor,yfloor],color="#9aa3b5",lw=1.8,zorder=6)
axB.text(8,yfloor-8,"floor",color="#9aa3b5",fontsize=10,ha="left")

joints={f"{sw}_ankle"}
for m,off,base,col,lab in ((mp,-10.5,[.29,.75,.53],"#4bbf87","PARAMETER blend"),
                           (mb,+10.5,[.90,.64,.34],"#e0a458","POSE blend")):
    ch=[(V+rv*off,F) for V,F in part_tris(m,m["euler"],m["ground"],joints)]
    shade(axB,PR2,ch,base,1.0,3)
    c=m["corners"][sw]; k=int(np.argmin(c[:,2])); pw=(C2R@c[k])*100
    pv=np.array([-pw[1],pw[0],pw[2]])+rv*off
    a2,_=PR2(pv)
    x=float(a2[0,0]); y=float(a2[0,1])
    axB.annotate("",xy=(x,yfloor),xytext=(x,y),
        arrowprops=dict(arrowstyle="<->",color=col,lw=2.2,shrinkA=0,shrinkB=0),zorder=9)
    axB.plot([x],[y],"o",color=col,ms=6,zorder=10)
    axB.text(x,yfloor+34,f"{m['clearance']*1000:+.2f} mm",color=col,fontsize=13.5,
             fontweight="bold",ha="center",va="center",zorder=10)
    axB.text(x,38 if m is mp else 62,lab,color=col,fontsize=11,fontweight="bold",
             ha="center",zorder=10)
axB.set_title("the same swing foot at the same instant, side by side",
              color="#c8cfdd",fontsize=10.5,pad=6)

# ---- clearance curves ----
axC=fig.add_subplot(G[0,2]); axC.set_facecolor("#161922")
for i,c in enumerate(CASES):
    ph=np.array([r[0] for r in c["rows"]])
    pc=np.array([r[1]["clearance"] for r in c["rows"]])*1000
    bc=np.array([r[2]["clearance"] for r in c["rows"]])*1000
    col=plt.cm.viridis(0.15+0.24*i)
    axC.plot(ph,bc,color=col,lw=1.9,label=c["short"])
    axC.plot(ph,pc,color=col,lw=1.0,ls=(0,(3,2)),alpha=.75)
axC.axhspan(-30,0,color="#e0655f",alpha=.10,zorder=0)
axC.axhline(0,color="#8a93a6",lw=1)
for x,nm in zip(GC.PHI_LANDMARKS,["LLO","LTD","RLO","RTD",""]):
    if nm: axC.axvline(x,color="#39405480",lw=.9,ls=":")
axC.set_xlim(0,1); axC.set_ylim(-13.5,9)
axC.set_xlabel("gait phase  φ",color="#828a9c",fontsize=9.5)
axC.set_ylabel("swing-foot clearance  [mm]",color="#828a9c",fontsize=9.5)
axC.set_title("solid = pose blend    dashed = parameter blend\n(y zoomed; the swing arc peaks off-scale at ~43 mm)",
              color="#c8cfdd",fontsize=10.5,pad=6)
axC.legend(fontsize=8.5,facecolor="#1a1d26",edgecolor="#2a2f3d",labelcolor="#c8cfdd")
axC.tick_params(colors="#828a9c",labelsize=8.5)
for s in axC.spines.values(): s.set_color("#2a2f3d")
axC.text(.02,.04,"below 0 = foot through the floor",transform=axC.transAxes,
         color="#e0655f",fontsize=8.5)

# ---- bottom: what is actually being interpolated ----
axD=fig.add_subplot(G[1,0]); axD.set_facecolor("#161922")
A,Bs=CASES[2]["A"],CASES[2]["B"]; bl=CASES[2]["blend"]
tt=np.linspace(0,1,300)
for s,lab,col,lw in ((A,A.name,"#e0655f",1.4),(Bs,Bs.name,"#5b8ff9",1.4)):
    axD.plot(tt,[s.c_foot_z["L"](s.phase(p))*100 for p in tt],color=col,lw=lw,label=lab)
axD.plot(tt,[bl.c_foot_z["L"](bl.phase(p))*100 for p in tt],color="#4bbf87",lw=2.4,
         label="blended curve")
axD.axhline(0,color="#8a93a6",lw=1)
for x in GC.PHI_LANDMARKS[1:4]: axD.axvline(x,color="#39405480",lw=.9,ls=":")
axD.set_xlim(0.055,0.145); axD.set_ylim(-0.35,1.6)
axD.axhspan(-0.35,0,color="#e0655f",alpha=.10,zorder=0)
axD.annotate("all three hit exactly 0\nat LTD (φ=0.1)",xy=(0.1,0),xytext=(0.108,0.75),
   color="#4bbf87",fontsize=9,arrowprops=dict(arrowstyle="->",color="#4bbf87",lw=1.1))
axD.set_xlabel("gait phase  φ   (zoomed on touchdown)",color="#828a9c",fontsize=9.5)
axD.set_ylabel("left-foot swing z curve  [cm]",color="#828a9c",fontsize=9.5)
axD.set_title("PARAMETER space: the curve is interpolated, and the\n"
              "clamped endpoint is preserved by any convex combination",
              color="#c8cfdd",fontsize=10.5,pad=6)
axD.legend(fontsize=8.5,facecolor="#1a1d26",edgecolor="#2a2f3d",labelcolor="#c8cfdd")
axD.tick_params(colors="#828a9c",labelsize=8.5)
for s in axD.spines.values(): s.set_color("#2a2f3d")

axE=fig.add_subplot(G[1,1]); axE.set_facecolor("#161922")
c=CASES[2]
ph=np.array([r[0] for r in c["rows"]])
for s,res,lab,col in ((c["A"],c["resA"],c["A"].name,"#e0655f"),
                      (c["B"],c["resB"],c["B"].name,"#5b8ff9")):
    v=[np.rad2deg(BC.sample_at_phi(res,s,p)[0]["L"][3]) for p in ph]
    axE.plot(ph,v,color=col,lw=1.4,label=lab)
vp=[np.rad2deg(r[1]["q"]["L"][3]) for r in c["rows"]]
vb=[np.rad2deg(r[2]["q"]["L"][3]) for r in c["rows"]]
axE.plot(ph,vb,color="#e0a458",lw=2.4,label="pose blend (mean)")
axE.plot(ph,vp,color="#4bbf87",lw=2.0,ls=(0,(4,2)),label="parameter blend (re-planned)")
for x in GC.PHI_LANDMARKS[1:4]: axE.axvline(x,color="#39405480",lw=.9,ls=":")
axE.set_xlim(0,1); axE.set_xlabel("gait phase  φ",color="#828a9c",fontsize=9.5)
axE.set_ylabel("left knee angle  [deg]",color="#828a9c",fontsize=9.5)
axE.set_title("JOINT space: averaging the angles is not the same\n"
              "as re-solving IK for the blended foot path",
              color="#c8cfdd",fontsize=10.5,pad=6)
axE.legend(fontsize=8.5,facecolor="#1a1d26",edgecolor="#2a2f3d",labelcolor="#c8cfdd")
axE.tick_params(colors="#828a9c",labelsize=8.5)
for s in axE.spines.values(): s.set_color("#2a2f3d")

axF=fig.add_subplot(G[1,2]); axF.set_facecolor("#161922")
names=[c["short"] for c in CASES]
pmin=[min(r[1]["clearance"] for r in c["rows"])*1000 for c in CASES]
bmin=[min(r[2]["clearance"] for r in c["rows"])*1000 for c in CASES]
y=np.arange(len(names))
axF.barh(y+.19,pmin,height=.34,color="#4bbf87",label="parameter blend")
axF.barh(y-.19,bmin,height=.34,color="#e0a458",label="pose blend")
for i,(p,b) in enumerate(zip(pmin,bmin)):
    axF.text(p-.4,i+.19,f"{p:+.2f}",va="center",ha="right",color="#4bbf87",fontsize=9)
    axF.text(b-.4,i-.19,f"{b:+.2f}",va="center",ha="right",color="#e0a458",fontsize=9)
axF.set_yticks(y); axF.set_yticklabels(names,color="#c8cfdd",fontsize=9)
axF.axvline(0,color="#8a93a6",lw=1)
axF.set_xlabel("worst swing-foot clearance over the cycle  [mm]",
               color="#828a9c",fontsize=9.5)
axF.set_title("parameter blending never breaks contact\n"
              "pose blending gets worse the more the samples differ",
              color="#c8cfdd",fontsize=10.5,pad=6)
axF.legend(fontsize=8.5,facecolor="#1a1d26",edgecolor="#2a2f3d",labelcolor="#c8cfdd",loc="lower left")
axF.tick_params(colors="#828a9c",labelsize=8.5)
for s in axF.spines.values(): s.set_color("#2a2f3d")

fig.suptitle("Blending gait PARAMETERS vs blending POSES — identical phase registration, "
             "identical weights, identical samples",
             color="#e6ebf2",fontsize=14,y=.972)
fig.text(.5,.030,"Parameter blending interpolates Π and re-runs the planner, so the contact "
   "constraint is re-imposed and the foot lands exactly on the floor at every blend.",
   color="#828a9c",fontsize=10.5,ha="center")
fig.text(.5,.008,"Pose blending interpolates the planner's output — and foot height is a "
   "nonlinear function of the joint angles, so nothing carries the constraint through.",
   color="#828a9c",fontsize=10.5,ha="center")
fig.savefig(os.path.join(HERE,"blend_param_vs_pose.png"),dpi=105,facecolor="#12141a")
print("wrote blend_param_vs_pose.png")
