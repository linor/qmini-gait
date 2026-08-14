/* =====================================================================
   param_step.js -- the real-time planner loop (paper Sec. 4.2).

   Unlike the offline generator, the command changes every frame, so the
   footstep plan is re-evaluated continuously during swing and latched at
   touchdown -- exactly as the paper describes.  The root is NOT solved from
   the planted foot here: the path frame IS the integral of the command, so the
   robot tracks the commanded velocity by construction and the feet are latched
   rather than inferred.
   ===================================================================== */

/* ---- Sec. 5.3 three-step weights (identical to the pose-blend viewer) ---- */
const FR=["reverse_fast","reverse_slow","step_in_place","forward_slow","forward_fast"]
        .map(n=>BY[n]);
const FRV=FR.map(s=>s.vel[0]);
const TS=["left_turn","left_strafe","right_turn","right_strafe"].map(n=>BY[n]);
const TSB=[0,Math.PI/2,Math.PI,3*Math.PI/2];
const XMAX=PARAMS.keyvel.xmax, YMAX=PARAMS.keyvel.ymax, THMAX=PARAMS.keyvel.thmax;
let ETA=0.5;

function blendWeights(v){
  const w=new Map(); SAMPLES.forEach(s=>w.set(s,0));
  const x=clamp(v[0],FRV[0],FRV[FRV.length-1]);
  let i=0; while(i<FRV.length-2&&x>FRV[i+1])i++;
  const aFR=f_soft((x-FRV[i])/(FRV[i+1]-FRV[i]));
  const yn=YMAX?v[1]/YMAX:0, tn=THMAX?v[2]/THMAX:0;
  const rho=Math.min(Math.hypot(yn,tn),1);
  let beta=Math.atan2(yn,tn); if(beta<0)beta+=2*Math.PI;
  const seg=Math.min(3,Math.floor(beta/(Math.PI/2)));
  const aB=f_soft((beta-TSB[seg])/(Math.PI/2));
  const xn=XMAX?Math.abs(v[0])/XMAX:0;
  const aTS=f_soft(rho)*(1-ETA*f_soft(xn));
  w.set(FR[i],  w.get(FR[i])  +(1-aTS)*(1-aFR));
  w.set(FR[i+1],w.get(FR[i+1])+(1-aTS)*aFR);
  w.set(TS[seg],        w.get(TS[seg])        +aTS*(1-aB));
  w.set(TS[(seg+1)%4],  w.get(TS[(seg+1)%4])  +aTS*aB);
  return {w,beta,rho,aTS,xn,yn,tn};
}

function forcedInfo(s){
  const w=new Map(); SAMPLES.forEach(x=>w.set(x,0)); w.set(s,1);
  return {w,beta:0,rho:0,aTS:0,xn:0,yn:0,tn:0,forced:true};
}

/* ---------------------------- live state ---------------------------- */
const S={
  phi:0, cmd:[0,0,0], tgt:[0,0,0],
  path:[0,0,0],
  latched:{L:null,R:null}, target:{L:null,R:null},
  contact:"DS2", B:null, wts:null, info:null, force:null,
  q:{L:[0,0,0,0,0],R:[0,0,0,0,0]},
  pel:[0,0,0], pelR:[[1,0,0],[0,1,0],[0,0,1]],
  foot:{L:null,R:null},
  ikErr:0, ikMax:0, clearance:0, clearMin:0, slip:0, slipMax:0,
  prints:[], travelled:[0,0,0], measured:[0,0,0],
};

function resetState(){
  S.phi=0; S.cmd=[0,0,0]; S.tgt=[0,0,0]; S.path=[0,0,0];
  S.latched={L:null,R:null}; S.target={L:null,R:null};
  S.q={L:[0,0,0,0,0],R:[0,0,0,0,0]};
  S.prints=[]; S.clearMin=0; S.slipMax=0; S.ikMax=0; S.measured=[0,0,0];
  S.contact="DS2";
  stepSim(0);
}

/* nominal foothold for the upcoming contact of `side` (Sec. 4.2, Eq. 4):
   sample the predicted path frame at the midpoint of that contact interval */
function upcomingFoothold(B,side,t){
  const mid = side==="L" ? 0.5*(B.t_ltd+B.T)
                         : 0.5*(B.t_rtd+B.t_rlo+B.T);
  const dtf=Math.max(mid-t,0);
  const pv=compose(S.path,pathPose(S.cmd,dtf));
  return compose(pv,stancePose(B,side));
}
const lerpPose=(a,b,al)=>[a[0]+(b[0]-a[0])*al, a[1]+(b[1]-a[1])*al,
                          a[2]+wrapPi(b[2]-a[2])*al];

/* ---------------------------- one frame ---------------------------- */
function stepSim(dt){
  /* command ramp (Sec. 7.1 style limiting) */
  const AR=[0.55,0.35,1.4];
  for(let i=0;i<3;i++){
    const d=S.tgt[i]-S.cmd[i], m=AR[i]*dt;
    S.cmd[i]+=Math.abs(d)<=m?d:Math.sign(d)*m;
  }
  /* weights -> blended sample.  The editor pins a single sample so an edit can
     be judged in isolation; otherwise the Sec. 5.3 interpolation runs. */
  const info=S.force?forcedInfo(S.force):blendWeights(S.cmd);
  S.info=info; S.wts=info.w;
  const active=[...info.w.entries()].filter(([,q])=>q>1e-4).sort((a,b)=>b[1]-a[1]);
  const B=blendSamples(active); S.B=B;

  /* advance phase through the blended phase map */
  let t=B.om(S.phi)+dt;
  if(t>=B.T)t-=B.T*Math.floor(t/B.T);
  S.phi=B.om.inv(t);
  const st=contactOf(S.phi);

  /* path frame = integral of the commanded velocity */
  const prevPath=S.path.slice();
  S.path=compose(S.path,pathPose(S.cmd,dt));

  /* ---- footstep plan ---- */
  for(const side of["L","R"]){
    const swinging = side==="L" ? (S.phi<PHI[1]) : (S.phi>=PHI[2]&&S.phi<PHI[3]);
    if(swinging){
      S.target[side]=upcomingFoothold(B,side,t);
      if(!S.latched[side])S.latched[side]=S.target[side].slice();
    }else{
      if(!S.latched[side])
        S.latched[side]=compose(S.path,stancePose(B,side));
      /* the instant the foot lands, fix the foothold for the new support phase */
      const justLanded=(side==="L"&&S.contact==="LS")||(side==="R"&&S.contact==="RS");
      if(justLanded&&S.target[side]){
        S.latched[side]=S.target[side].slice();
        S.prints.push({side,pose:S.latched[side].slice()});
        if(S.prints.length>64)S.prints.shift();
      }
    }
  }

  /* ---- foot poses: nominal + stylized (Eq. 3) ---- */
  const poses={};
  for(const side of["L","R"]){
    const a=side==="L"?0:B.t_rlo, b=side==="L"?B.t_ltd:B.t_rtd;
    const swinging=(t>=a&&t<b);
    let base,baseYaw,dx=0,dy=0,dz=0,dyaw=0,dgam=0,ph=0,pt=0;
    if(swinging){
      const al=f_soft((t-a)/(b-a));
      const p=lerpPose(S.latched[side],S.target[side]||S.latched[side],al);
      base=[p[0],p[1],0]; baseYaw=p[2];
      dx=B.curve("foot_x",side)(t); dy=B.curve("foot_y",side)(t);
      dz=B.curve("foot_z",side)(t); dyaw=B.curve("foot_yaw",side)(t);
      dgam=B.curve("foot_gamma",side)(t);
      ph=B.curve("psi_heel",side)(t); pt=B.curve("psi_toe",side)(t);
    }else{
      const L=S.latched[side];
      base=[L[0],L[1],0]; baseYaw=L[2];
      /* automated heel/toe transitions during support (Sec. 4.3, Attributes) */
      const tTD=side==="L"?B.t_ltd:B.t_rtd;
      const tLO=side==="L"?B.T:B.t_rlo+B.T;
      const te=t>=tTD?t:t+B.T;
      ph=rampDown(te,tTD,B.dt_psi_td,B.psi_heel_td);
      pt=rampUp(te,tLO,B.dt_psi_lo,B.psi_toe_lo);
    }
    const [Rht,tht]=heelToe(ph,pt);
    const Rl=mul33(rotZYX(dyaw,0,dgam),Rht);
    const pl=add3([dx,dy,dz],tht);
    const Rb=rotZ(baseYaw);
    poses[side]={p:add3(base,mul3(Rb,pl)), R:mul33(Rb,Rl), contact:!swinging};
  }

  /* ---- pelvis (Sec. 4.2, nominal + stylized) ---- */
  const pz=B.pelvis_z_nom+B.curve("pelvis_z")(t);
  const yaw=S.path[2]+B.curve("pelvis_yaw")(t);
  const pitch=B.curve("pelvis_pitch")(t), roll=B.curve("pelvis_roll")(t);
  const off=mul3(rotZ(S.path[2]),[KIN.pelvis_x_offset,0,0]);
  const pPel=[S.path[0]+off[0],S.path[1]+off[1],pz];
  const RPel=rotZYX(yaw,pitch,roll);
  S.pel=pPel; S.pelR=RPel;

  /* ---- IK, with the contact-height correction ---- */
  let ikErr=0, clearance=Infinity;
  for(const side of["L","R"]){
    const fp=poses[side];
    const RPelT=[[RPel[0][0],RPel[1][0],RPel[2][0]],
                 [RPel[0][1],RPel[1][1],RPel[2][1]],
                 [RPel[0][2],RPel[1][2],RPel[2][2]]];
    const [yawT,pitchT]=yawPitchOf(mul33(RPelT,fp.R));
    let lift=0, z0=fp.p[2], q=S.q[side], err=0;
    for(let k=0;k<3;k++){
      const tgt=[fp.p[0],fp.p[1],z0+lift];
      const r=LEGS[side].ik(tgt,yawT,pitchT,q,pPel,RPel);
      q=r[0]; err=r[1];
      const [,Rach]=LEGS[side].fkSole(q,pPel,RPel);
      let low=Infinity;
      for(const c of LEGS[side].corners) low=Math.min(low,mul3(Rach,c)[2]);
      const nl=Math.max(0,-z0-low);
      if(Math.abs(nl-lift)<1e-7)break;
      lift=nl;
    }
    S.q[side]=q; ikErr=Math.max(ikErr,err);
    const [sole,Rach]=LEGS[side].fkSole(q,pPel,RPel);
    let low=Infinity;
    for(const c of LEGS[side].corners) low=Math.min(low,sole[2]+mul3(Rach,c)[2]);
    S.foot[side]={p:sole,R:Rach,contact:fp.contact,low};
    if(!fp.contact) clearance=Math.min(clearance,low);
  }
  S.ikErr=ikErr; S.ikMax=Math.max(S.ikMax,ikErr);
  S.clearance=isFinite(clearance)?clearance:0;
  S.clearMin=Math.min(S.clearMin,S.clearance);

  /* diagnostics: planted-foot slip and the velocity actually travelled */
  let slip=0;
  for(const side of["L","R"]){
    const f=S.foot[side];
    if(f.contact&&S._prev&&S._prev[side]&&S._prev[side].contact)
      slip=Math.max(slip,Math.hypot(f.p[0]-S._prev[side].p[0],
                                    f.p[1]-S._prev[side].p[1]));
  }
  S.slip=slip; if(dt>0)S.slipMax=Math.max(S.slipMax,slip);
  S._prev={L:{...S.foot.L},R:{...S.foot.R}};
  if(dt>0){
    const dx=S.path[0]-prevPath[0], dy=S.path[1]-prevPath[1];
    const c=Math.cos(-prevPath[2]), s=Math.sin(-prevPath[2]);
    const bx=(c*dx-s*dy)/dt, by=(s*dx+c*dy)/dt;
    const k=Math.min(1,dt*3);
    const dth=wrapPi(S.path[2]-prevPath[2])/dt;
    S.measured=[S.measured[0]+(bx-S.measured[0])*k,
                S.measured[1]+(by-S.measured[1])*k,
                S.measured[2]+(dth-S.measured[2])*k];
  }
  S.contact=st;
}
