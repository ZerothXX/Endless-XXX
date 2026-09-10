const canvas=document.querySelector('.cover-particles'),ctx=canvas.getContext('2d');
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const points=Array.from({length:55},(_,i)=>({x:((i*997)%1000)/1000,y:((i*613)%1000)/1000,s:.3+(i%5)*.12}));
let width=0,height=0;
function resize(){width=innerWidth;height=innerHeight;const dpr=Math.min(devicePixelRatio,2);canvas.width=width*dpr;canvas.height=height*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);}
function frame(time){if(!document.querySelector('#view-cover').hidden){ctx.clearRect(0,0,width,height);const positions=points.map((p,i)=>({x:((p.x+(reduced?0:time*.000006*p.s))%1)*width,y:(p.y+Math.sin((reduced?0:time*.0002)+i)*.015)*height}));positions.forEach((p,i)=>{ctx.fillStyle=i%4===0?'#95deff':'#408fff';ctx.globalAlpha=.25+points[i].s*.4;ctx.beginPath();ctx.arc(p.x,p.y,i%6===0?2:1,0,Math.PI*2);ctx.fill();for(let j=i+1;j<positions.length;j++){const q=positions[j],distance=Math.hypot(q.x-p.x,q.y-p.y);if(distance<140){ctx.strokeStyle='#58aaff';ctx.globalAlpha=(1-distance/140)*.16;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();}}});ctx.globalAlpha=1;}requestAnimationFrame(frame);}
window.addEventListener('resize',resize);resize();requestAnimationFrame(frame);
