/* =====================================================================
   param_core.js -- PARAMETER blending, live.

   Interpolate the gait sample Pi (Tab. 1 time intervals, Sec. 4.3 attributes,
   Tab. 2 function curves) into a new valid sample, then re-run the paper's
   planner (Sec. 4.2: footstep plan, nominal support/swing trajectories,
   heel-toe rig) and solve the leg IK.  Nothing is a baked pose.

   Injected by the page: PARAMS (gait_params.json).
   ===================================================================== */

const PHI = PARAMS.phi;                       // [0, .1, .5, .6, 1]
const KIN = PARAMS.kin;
const CHAIN = KIN.chain;
const ANKLE_IN_FOOT = KIN.ankle_in_foot;
const HEEL_IN_FOOT  = KIN.heel_in_foot;
const TOE_IN_FOOT   = KIN.toe_in_foot;

/* ---------------------------- small math ---------------------------- */
const clamp=(v,a,b)=>v<a?a:(v>b?b:v);
const wrapPi=a=>Math.atan2(Math.sin(a),Math.cos(a));
const add3=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const sub3=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const mul3=(m,v)=>[m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2],
                   m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],
                   m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]];
const mul33=(a,b)=>[0,1,2].map(i=>[0,1,2].map(j=>
   a[i][0]*b[0][j]+a[i][1]*b[1][j]+a[i][2]*b[2][j]));
function rotAxis(ax,ang){
  const l=Math.hypot(ax[0],ax[1],ax[2])||1, x=ax[0]/l,y=ax[1]/l,z=ax[2]/l;
  const c=Math.cos(ang),s=Math.sin(ang),C=1-c;
  return [[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
          [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
          [z*x*C-y*s, z*y*C+x*s, c+z*z*C]];
}
const rotX=a=>rotAxis([1,0,0],a), rotY=a=>rotAxis([0,1,0],a), rotZ=a=>rotAxis([0,0,1],a);
const rotZYX=(y,p,r)=>mul33(mul33(rotZ(y),rotY(p)),rotX(r));
function yawPitchOf(R){
  const pitch=-Math.asin(clamp(R[2][0],-1,1));
  return [Math.atan2(R[1][0],R[0][0]),pitch];
}
const f_soft=a=>{a=clamp(a,0,1);return a*a*(3-2*a);};      // Eq. 8

/* ---------------------------- splines ---------------------------- */
function hermite(sp){
  const t=sp.t,v=sp.v,m=sp.m,n=t.length,per=sp.periodic,T=t[n-1]-t[0];
  return function(x){
    if(per) x=t[0]+((x-t[0])%T+T)%T;
    const xc=clamp(x,t[0],t[n-1]);
    let i=0; while(i<n-2&&xc>t[i+1])i++;
    const h=t[i+1]-t[i],s=(xc-t[i])/h,s2=s*s,s3=s2*s;
    return (2*s3-3*s2+1)*v[i]+(s3-2*s2+s)*h*m[i]
         +(-2*s3+3*s2)*v[i+1]+(s3-s2)*h*m[i+1];
  };
}
/* monotone Fritsch-Carlson, used only for the phase map Omega(phi) */
function monoSpline(xs,ys){
  const n=xs.length,d=[],m=new Array(n);
  for(let i=0;i<n-1;i++)d.push((ys[i+1]-ys[i])/(xs[i+1]-xs[i]));
  m[0]=d[0]; m[n-1]=d[n-2];
  for(let i=1;i<n-1;i++)m[i]=0.5*(d[i-1]+d[i]);
  for(let i=0;i<n-1;i++){
    if(d[i]===0){m[i]=0;m[i+1]=0;continue;}
    const a=m[i]/d[i],b=m[i+1]/d[i],s=a*a+b*b;
    if(s>9){const tau=3/Math.sqrt(s);m[i]=tau*a*d[i];m[i+1]=tau*b*d[i];}
  }
  const f=x=>{
    const xc=clamp(x,xs[0],xs[n-1]);
    let i=0; while(i<n-2&&xc>xs[i+1])i++;
    const h=xs[i+1]-xs[i],s=(xc-xs[i])/h,s2=s*s,s3=s2*s;
    return (2*s3-3*s2+1)*ys[i]+(s3-2*s2+s)*h*m[i]
         +(-2*s3+3*s2)*ys[i+1]+(s3-s2)*h*m[i+1];
  };
  f.inv=y=>{let lo=xs[0],hi=xs[n-1];
    for(let k=0;k<48;k++){const mid=.5*(lo+hi); if(f(mid)<y)lo=mid;else hi=mid;}
    return .5*(lo+hi);};
  return f;
}

/* ---------------------------- samples ---------------------------- */
class Sample{
  constructor(d){
    Object.assign(this,d);
    this.events=[0,d.t_ltd,d.t_rlo,d.t_rtd,d.T];
    this.om=monoSpline(PHI,this.events);
    this.c={};
    for(const k in d.curves){
      const cv=d.curves[k];
      this.c[k]=cv.L?{L:hermite(cv.L),R:hermite(cv.R)}:hermite(cv);
    }
  }
}
const SAMPLES=PARAMS.samples.map(d=>new Sample(d));
const BY={}; SAMPLES.forEach(s=>BY[s.name]=s);

/* -------- Sec. 5.2: a convex combination of samples is a valid sample -------- */
function blendSamples(active){          // active = [[Sample, w], ...], sum w = 1
  const B={srcs:active};
  const mix=f=>{let o=0; for(const[s,w]of active)o+=w*f(s); return o;};
  // Tab. 1 intervals: alpha-blend the limits; a convex combination cannot
  // reorder them, so t_LLO < t_LTD < t_RLO < t_RTD holds automatically
  B.t_llo=0;
  B.t_ltd=mix(s=>s.t_ltd); B.t_rlo=mix(s=>s.t_rlo); B.t_rtd=mix(s=>s.t_rtd);
  B.T=mix(s=>s.T);
  B.events=[0,B.t_ltd,B.t_rlo,B.t_rtd,B.T];
  B.om=monoSpline(PHI,B.events);
  // attributes: Eq. 7
  for(const k of["stance_width","stance_x_bias","stance_splay","dt_psi_lo",
                 "dt_psi_td","psi_toe_lo","psi_heel_td","pelvis_z_nom"])
    B[k]=mix(s=>s[k]);
  // function curves: registered blend -- map t -> phi with the BLENDED phase
  // map, then back into each source's own cycle time
  B.curve=(nm,side)=>t=>{
    const phi=B.om.inv(t);
    let o=0;
    for(const[s,w]of active){
      const ts=s.om(phi);
      o+=w*(side?s.c[nm][side](ts):s.c[nm](ts));
    }
    return o;
  };
  return B;
}

/* ---------------------------- kinematics ---------------------------- */
class Leg{
  constructor(side){
    const k=KIN.legs[side];
    this.side=side; this.a=k.anchor; this.local=k.local; this.ax=k.axis;
    this.lim=k.limits; this.corners=k.corners;
  }
  fk(q,pPel,RPel){
    let R=RPel, p=add3(pPel,mul3(RPel,this.a.hip_yaw));
    R=mul33(R,rotAxis(this.ax.hip_yaw,q[0]));
    for(let i=1;i<CHAIN.length;i++){
      const j=CHAIN[i];
      p=add3(p,mul3(R,this.local[j]));
      R=mul33(R,rotAxis(this.ax[j],q[i]));
    }
    return [p,R];
  }
  /* sole reference point, not the ankle: this robot has no ankle roll, so an
     ankle-based target drags the contact point sideways under hip roll */
  fkSole(q,pPel,RPel){
    const [p,R]=this.fk(q,pPel,RPel);
    return [add3(p,mul3(R,[-ANKLE_IN_FOOT[0],-ANKLE_IN_FOOT[1],-ANKLE_IN_FOOT[2]])),R];
  }
  residual(q,pT,yawT,pitchT,pPel,RPel){
    const [p,R]=this.fkSole(q,pPel,RPel);
    const RL=mul33([[RPel[0][0],RPel[1][0],RPel[2][0]],
                    [RPel[0][1],RPel[1][1],RPel[2][1]],
                    [RPel[0][2],RPel[1][2],RPel[2][2]]],R);
    const [yaw,pitch]=yawPitchOf(RL);
    return [pT[0]-p[0],pT[1]-p[1],pT[2]-p[2],
            wrapPi(yawT-yaw),wrapPi(pitchT-pitch)];
  }
  ik(pT,yawT,pitchT,q0,pPel,RPel,iters=32){
    const W=[1,1,1,.12,.12], h=1e-6, lam=1e-4;
    let q=q0.map((v,i)=>clamp(v,this.lim[i][0],this.lim[i][1]));
    let best=q.slice(), bestC=Infinity;
    for(let it=0;it<iters;it++){
      const e=this.residual(q,pT,yawT,pitchT,pPel,RPel);
      let c=0; for(let i=0;i<5;i++)c+=(e[i]*W[i])**2; c=Math.sqrt(c);
      if(c<bestC){bestC=c;best=q.slice();}
      if(c<1e-9)break;
      const J=[];
      for(let k=0;k<5;k++){
        const qd=q.slice(); qd[k]+=h;
        const e2=this.residual(qd,pT,yawT,pitchT,pPel,RPel);
        J.push([0,1,2,3,4].map(r=>-(e2[r]-e[r])/h));      // column k
      }
      // normal equations on the weighted system (5x5)
      const A=[],b=[];
      for(let i=0;i<5;i++){
        A.push(new Array(5).fill(0)); b.push(0);
        for(let j=0;j<5;j++){
          let s=0; for(let r=0;r<5;r++)s+=J[i][r]*W[r]*J[j][r]*W[r];
          A[i][j]=s+(i===j?lam:0);
        }
        let s=0; for(let r=0;r<5;r++)s+=J[i][r]*W[r]*e[r]*W[r];
        b[i]=s;
      }
      const dq=solve5(A,b);
      if(!dq)break;
      let step=Math.hypot(...dq);
      const sc=step>0.20?0.20/step:1;
      const qn=q.map((v,i)=>clamp(v+dq[i]*sc,this.lim[i][0],this.lim[i][1]));
      let mx=0; for(let i=0;i<5;i++)mx=Math.max(mx,Math.abs(qn[i]-q[i]));
      q=qn;
      if(mx<1e-10)break;
    }
    const e=this.residual(best,pT,yawT,pitchT,pPel,RPel);
    return [best,Math.hypot(e[0],e[1],e[2])];
  }
}
function solve5(A,b){
  const n=5,M=A.map((r,i)=>r.concat([b[i]]));
  for(let c=0;c<n;c++){
    let p=c; for(let r=c+1;r<n;r++)if(Math.abs(M[r][c])>Math.abs(M[p][c]))p=r;
    if(Math.abs(M[p][c])<1e-14)return null;
    [M[c],M[p]]=[M[p],M[c]];
    for(let r=0;r<n;r++){
      if(r===c)continue;
      const f=M[r][c]/M[c][c];
      for(let k=c;k<=n;k++)M[r][k]-=f*M[c][k];
    }
  }
  return M.map((r,i)=>r[n]/M[i][i]);
}
const LEGS={L:new Leg("L"),R:new Leg("R")};

/* ---------------------------- path frame ---------------------------- */
function pathPose(vel,t){                       // Sec. 4.2 constant-velocity arc
  const [xd,yd,td]=vel, th=td*t;
  if(Math.abs(td)<1e-9)return [xd*t,yd*t,th];
  const s=Math.sin(th),c=Math.cos(th);
  return [(xd*s-yd*(1-c))/td,(xd*(1-c)+yd*s)/td,th];
}
function compose(pa,pb){
  const c=Math.cos(pa[2]),s=Math.sin(pa[2]);
  return [pa[0]+c*pb[0]-s*pb[1], pa[1]+s*pb[0]+c*pb[1], pa[2]+pb[2]];
}
function stancePose(B,side){
  const sg=side==="L"?1:-1;
  return [KIN.stance_x_neutral+B.stance_x_bias, sg*0.5*B.stance_width, sg*B.stance_splay];
}

/* heel/toe rig (Fig. 5): rotate about the heel, then about the resulting toe */
function heelToe(psiHeel,psiToe){
  const R1=rotY(psiHeel);
  const t1=sub3(HEEL_IN_FOOT,mul3(R1,HEEL_IN_FOOT));
  const toe1=add3(mul3(R1,TOE_IN_FOOT),t1);
  const R2=rotY(psiToe);
  const t2=sub3(toe1,mul3(R2,toe1));
  return [mul33(R2,R1), add3(mul3(R2,t1),t2)];
}
const rampDown=(t,te,dt,v)=>{ if(dt<=0||v===0)return 0;
  const u=(t-te)/dt; return u<=0?v:u>=1?0:v*(1-u); };
const rampUp=(t,te,dt,v)=>{ if(dt<=0||v===0)return 0;
  const u=(t-(te-dt))/dt; return u<=0?0:u>=1?v:v*u; };

const contactOf=phi=>phi<PHI[1]?"LS":phi<PHI[2]?"DS1":phi<PHI[3]?"RS":"DS2";
