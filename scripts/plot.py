import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, glob, os
from stylized_gait import verify as V
from stylized_gait import paths

OUT = paths.CLIPS
files=sorted(glob.glob(OUT+"/*.usda"))
JN=["hip_yaw","hip_roll","hip_pitch","knee","ankle"]
COL=dict(zip(JN,["#4c72b0","#dd8452","#55a868","#c44e52","#8172b3"]))

fig,axes=plt.subplots(3,9,figsize=(30,10.5))
for c,f in enumerate(files):
    d=V.parse(f); n=len(d["rots"])-1; fps=d["fps"]
    tn=np.arange(n+1)/n
    ax=axes[0,c]
    for jn in JN:
        k=d["joints"].index([j for j in d["joints"] if j.endswith("L_"+jn)][0])
        ang=[]
        for i in range(n+1):
            q=d["rots"][i][k]; a=2*np.degrees(np.arctan2(np.linalg.norm(q[1:]),q[0]))
            axis=q[1:]/(np.linalg.norm(q[1:])+1e-12)
            ref={"hip_yaw":[0,0,1],"hip_roll":[0,-1,0],"hip_pitch":[-1,0,0],
                 "knee":[-1,0,0],"ankle":[-1,0,0]}[jn]
            ang.append(a*np.sign(np.dot(axis,ref) or 1))
        ax.plot(tn,ang,color=COL[jn],lw=1.6,label=jn)
    for te,nm in zip(d["events"][:4],["LLO","LTD","RLO","RTD"]):
        ax.axvline(te/d["T"],color="0.75",lw=.8,ls="--")
        ax.text(te/d["T"],ax.get_ylim()[1],nm,fontsize=6,color="0.45",va="bottom",ha="center")
    ax.set_title(os.path.basename(f).replace(".usda",""),fontsize=9)
    ax.set_ylim(-55,55); ax.grid(alpha=.25)
    if c==0: ax.set_ylabel("left leg joints [deg]"); ax.legend(fontsize=6,ncol=2)

    axL=axes[1,c]
    aj={s:[j for j in d["joints"] if j.endswith(f"{s}_ankle")][0] for s in "LR"}
    for s_,col in (("L","#c44e52"),("R","#4c72b0")):
        z=[]
        for i in range(n+1):
            T=V.fk_frame(d,i); Rw,tw=T[aj[s_]]
            z.append(min((tw+Rw@c_)[2] for c_ in V.CORNERS_CM[s_]))
        axL.plot(tn,z,color=col,lw=1.6,label=f"{s_} foot")
    axL.axhline(0,color="k",lw=.8)
    axL.fill_between(tn,-1,0,color="0.9")
    axL.set_ylim(-1,7); axL.grid(alpha=.25)
    if c==0: axL.set_ylabel("lowest sole point [cm]"); axL.legend(fontsize=6)

    axP=axes[2,c]
    axP.plot(tn,d["phase"],color="#937860",lw=1.8,label="phase $\\phi$")
    axP.plot(np.array(d["om_t"])/d["T"],d["om_phi"],"k:",lw=1,label="$\\Omega^{-1}$")
    axP.scatter([t/d["T"] for t in d["events"]],d["phis"],c="#c44e52",zorder=5,s=22)
    for te,pe,nm in zip(d["events"],d["phis"],["LLO","LTD","RLO","RTD","END"]):
        axP.annotate(f"{nm}\n{pe:g}",(te/d["T"],pe),fontsize=6,color="#c44e52",
                     xytext=(3,-9),textcoords="offset points")
    st=d["state"]
    for i in range(n):
        axP.axvspan(tn[i],tn[i+1],ymin=0,ymax=0.06,
                    color={"LS":"#c44e52","DS1":"#cccccc","RS":"#4c72b0","DS2":"#cccccc"}[st[i]],lw=0)
    axP.set_ylim(-0.05,1.05); axP.grid(alpha=.25); axP.set_xlabel("t / T")
    if c==0: axP.set_ylabel("gait phase $\\phi$"); axP.legend(fontsize=6,loc="upper left")

fig.suptitle("QMini reference gait samples -- Tab. 3 velocities, phase-tagged for blending "
             "(bottom strip: red=left swing, blue=right swing, grey=double support)",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(OUT+"/gait_reference_overview.png",dpi=110)
print("wrote overview")
