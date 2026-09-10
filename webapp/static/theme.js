export function rgb(hex) { return hex.replace('#','').match(/../g).map(v=>parseInt(v,16)/255); }
export function hsl(hex) { const [r,g,b]=rgb(hex),max=Math.max(r,g,b),min=Math.min(r,g,b),d=max-min,l=(max+min)/2; let h=0,s=0;if(d){s=d/(1-Math.abs(2*l-1));h=max===r?((g-b)/d+6)%6:max===g?(b-r)/d+2:(r-g)/d+4;h*=60;}return [h,s*100,l*100]; }
const color=(h,s,l,a=1)=>`hsla(${h.toFixed(2)}, ${Math.min(100,Math.max(0,s)).toFixed(2)}%, ${Math.min(100,Math.max(0,l)).toFixed(2)}%, ${a})`;
export const DEFAULT={name:'',mode:'dark',palette:[{hex:'#174fba'},{hex:'#dceeff'},{hex:'#986bff'}],assets:{}};
export function buildThemeVars(palette, mode, assets={}) {
 const [h,s]=hsl(palette[0].hex),[a,as,al]=hsl(palette[2].hex),light=mode==='light';
 const ring=color(a,as,light?al:Math.min(al+15,82));
 const values={
 '--bg0':color(h,light?Math.min(s,45):s,light?94:12),'--bg1':color(h,light?Math.min(s,50):s,light?87:17),
 '--bg-mid':color(h,s,light?76:22),'--bg2':color(h,Math.min(s+6,60),light?64:32),
 '--deep-pool':color(h,light?Math.min(s+10,65):s,light?46:9),
 '--surface':light?'rgba(255,255,255,.60)':color(h,75,28,.65),
 '--ink':color(h,light?55:30,light?22:95),'--ink-soft':color(h,light?38:30,light?34:78),
 '--line':color(light?h:(h+340)%360,s,light?58:68,.8),'--panel-ring':ring,'--ambient':color(a,as,al,.08),
 '--bb-accent':ring,'--bb-primary':color(h,s,light?58:57),'--glow':color(h,s,light?70:57,.40),
 '--cloud-hi':color(h,s,light?95:60,.72),'--cloud-mid':color(h,s,light?80:32,.55),
 '--cloud-low':color(h,s,light?76:38,.60),'--highlight':color(h,30,98,.90),
 '--screen':color(h,s,light?88:29,.75),'--button':color(h,s,light?73:29,.75),
 '--shadow':color(h,65,light?34:2,.27),'--log-bg':color(h,40,light?91:9,.96),
 '--muted':color(h,20,light?55:55,.48),'--scrim':color(h,40,light?25:5,.35),
 '--white-glint':color(h,35,98,.72),'--pm-asset':'url("/themes/shared/pm.png")',
 '--bb-asset':'url("/themes/shared/bb.png")','--accent-opacity':assets.character?'.78':'1',
 '--mascot-primary':assets.character?ring:color(h,85,58)
 }; return values;
}
export function applyTheme(root,theme) {const vars=buildThemeVars(theme.palette,theme.mode,{...theme.assets,character:!!theme.name});Object.entries(vars).forEach(([k,v])=>root.style.setProperty(k,v));root.dataset.theme=theme.name||'default';root.dataset.mode=theme.mode;return vars;}
function hash(text){let h=2166136261;for(const c of text) h=Math.imul(h^c.charCodeAt(0),16777619);return h>>>0;}
function rng(seed){return ()=>{let t=seed+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;};}
export function ornaments(name,width=1600){const random=rng(hash(name));const anchors=[[3,18],[6,46],[57,9],[94,27],[91,54],[60,82],[41,91],[88,86],[28,8],[97,73],[18,91],[70,11]];return anchors.slice(0,width<700?6:10).map(([x,y])=>({x,y,transform:`translate(-50%,-50%) rotate(${(-40+80*random()).toFixed(2)}deg) skewX(${(-12+24*random()).toFixed(2)}deg) scale(${(.5+.9*random()).toFixed(2)},${(.85+.4*random()).toFixed(2)})`,opacity:.75+.25*random()}));}
export function scatter(root,theme){const layer=root.querySelector('.accessories');layer.replaceChildren();if(!theme.assets.accessory)return;for(const a of ornaments(theme.name,innerWidth)){const el=document.createElement('span');el.className='accessory';Object.assign(el.style,{left:a.x+'%',top:a.y+'%',transform:a.transform,opacity:a.opacity,backgroundImage:`url("${theme.assets.accessory}")`});layer.append(el);}}
export function auditTheme(root,theme){const actual=getComputedStyle(root),expected=buildThemeVars(theme.palette,theme.mode,{...theme.assets,character:!!theme.name});const [h]=hsl(theme.palette[0].hex);const vars=Object.keys(expected),backgrounds=['--bg0','--bg1','--bg-mid','--bg2','--deep-pool'];const pm=root.querySelector('.pm');const report={variablesComplete:vars.every(k=>root.style.getPropertyValue(k)===expected[k]),backgroundHue:backgrounds.every(k=>Math.abs(parseFloat(expected[k].match(/[\d.]+/)[0])-h)<.1),ambientAlpha:expected['--ambient'].endsWith('0.08)'),pmUnchanged:getComputedStyle(pm).filter==='none'&&actual.getPropertyValue('--pm-asset').includes('/shared/pm.png'),noLeak:document.querySelector('#view-train').dataset.theme==='default',accessoryCount:root.querySelectorAll('.accessory').length,deterministic:JSON.stringify(ornaments(theme.name))===JSON.stringify(ornaments(theme.name)),writtenVariables:vars.length,changedValues:vars.filter(k=>expected[k]!==buildThemeVars(DEFAULT.palette,DEFAULT.mode)[k]).length};console.table(report);return report;}
