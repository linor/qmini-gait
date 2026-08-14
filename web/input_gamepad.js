/* =====================================================================
   input_gamepad.js -- keyboard + gamepad command input, shared by both
   interactive viewers.

   Standard-mapping layout:
     left stick    ẋ forward / ẏ strafe   (analog, so the blend space is
                                            swept continuously)
     right stick X θ̇ turn
     A / cross     toggle slow <-> fast
     B / circle    stop
     start / back  reset
     D-pad         digital equivalent of the left stick + turn on left/right

   Keyboard stays live at the same time; whichever axis is pushed further wins,
   so you can mix. Holding shift forces fast regardless of the toggle.
   ===================================================================== */

const IN={pad:null,padId:"",fast:false,lx:0,ly:0,rx:0,btn:[],prev:[],just:[],
          seen:false};

addEventListener("gamepadconnected",e=>{IN.padId=e.gamepad.id;IN.seen=true;});
addEventListener("gamepaddisconnected",()=>{IN.pad=null;IN.padId="";});

function dz2(x,y,dz){                       // radial deadzone, rescaled
  const m=Math.hypot(x,y);
  if(m<dz)return[0,0];
  const s=(m-dz)/(1-dz)/m;
  return[clamp(x*s,-1,1),clamp(y*s,-1,1)];
}
function dz1(x,dz){
  const a=Math.abs(x);
  if(a<dz)return 0;
  return Math.sign(x)*(a-dz)/(1-dz);
}

function pollPad(){
  const pads=navigator.getGamepads?navigator.getGamepads():[];
  let p=null;
  for(const g of pads) if(g&&g.connected){p=g;break;}
  IN.pad=p;
  if(!p){IN.lx=IN.ly=IN.rx=0;IN.btn=[];IN.just=[];IN.prev=[];return;}
  IN.padId=p.id; IN.seen=true;
  const a=p.axes||[];
  const [lx,ly]=dz2(a[0]||0,a[1]||0,0.12);
  IN.lx=lx; IN.ly=ly;
  IN.rx=dz1(a[2]||0,0.14);
  const pressed=(p.buttons||[]).map(b=>b.pressed);
  IN.just=pressed.map((v,i)=>v&&!IN.prev[i]);
  IN.prev=pressed; IN.btn=pressed;
  if(IN.just[0]) setFast(!IN.fast);                 // A / cross
  if(IN.just[1]) padStop();                         // B / circle
  if(IN.just[8]||IN.just[9]) padReset();            // back / start
}

/* per-axis slow scaling, matching the original keyboard feel */
const SLOW=[0.5,0.6,0.55];

function inputCommand(shiftHeld){
  pollPad();
  const fast=IN.fast||shiftHeld;
  let x=0,y=0,t=0;
  if(keys.has("w"))x+=1; if(keys.has("s"))x-=1;
  if(keys.has("a"))y+=1; if(keys.has("d"))y-=1;
  if(keys.has("q"))t+=1; if(keys.has("e"))t-=1;
  /* analog sticks: stick up = forward, stick left = strafe left / turn left */
  const gx=-IN.ly, gy=-IN.lx, gt=-IN.rx;
  if(Math.abs(gx)>Math.abs(x))x=gx;
  if(Math.abs(gy)>Math.abs(y))y=gy;
  if(Math.abs(gt)>Math.abs(t))t=gt;
  /* D-pad */
  const b=IN.btn;
  if(b[12])x=Math.max(x,1); if(b[13])x=Math.min(x,-1);
  if(b[14])t=Math.max(t,1); if(b[15])t=Math.min(t,-1);
  const s=fast?[1,1,1]:SLOW;
  return [clamp(x,-1,1)*XMAX*s[0],
          clamp(y,-1,1)*YMAX*s[1],
          clamp(t,-1,1)*THMAX*s[2]];
}

/* ---------------- little stick / button read-out ---------------- */
function drawPad(){
  const c=document.getElementById("pad"); if(!c)return;
  const dpr=Math.min(devicePixelRatio||1,2), r=c.getBoundingClientRect();
  c.width=r.width*dpr; c.height=r.height*dpr;
  const g=c.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0);
  const w=r.width,h=r.height;
  g.clearRect(0,0,w,h);
  const R=16, cy=h/2-3;
  const stick=(cx,px,py,lab)=>{
    g.strokeStyle="#2f3646"; g.lineWidth=1;
    g.beginPath(); g.arc(cx,cy,R,0,7); g.stroke();
    g.beginPath(); g.moveTo(cx-R,cy); g.lineTo(cx+R,cy);
    g.moveTo(cx,cy-R); g.lineTo(cx,cy+R); g.stroke();
    g.fillStyle=IN.pad?"#6ea8fe":"#39405a";
    g.beginPath(); g.arc(cx+px*R*0.86,cy+py*R*0.86,4,0,7); g.fill();
    g.fillStyle="#828a9c"; g.font="8px ui-sans-serif"; g.textAlign="center";
    g.fillText(lab,cx,h-1);
  };
  stick(R+4, IN.lx, IN.ly, "move");
  stick(R*3+16, IN.rx, 0, "turn");
  g.textAlign="left"; g.font="9px ui-sans-serif";
  g.fillStyle=IN.pad?"#4bbf87":"#5a6270";
  const nm=IN.pad?(IN.padId.split("(")[0].trim()||"gamepad").slice(0,20)
                 :(IN.seen?"pad idle":"no gamepad");
  g.fillText(nm,R*4+26,cy-4);
  g.fillStyle=IN.fast?"#e0a458":"#5a6270";
  g.fillText(IN.fast?"FAST  (A toggles)":"slow  (A toggles)",R*4+26,cy+9);
}
