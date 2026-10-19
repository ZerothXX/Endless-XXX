const $ = (selector) => document.querySelector(selector);
const els = {
  cover: $('#cover'), chat: $('#chat'), title: $('#chat-title'),
  messages: $('#messages'), roles: $('#role-list'), previews: $('#previews'),
  description: $('#description'), input: $('#image-input'), send: $('#send'),
  train: $('#start-train'), tab: $('#input-tab'), mode: $('#mode-button'),
  modeMenu: $('#mode-menu'), toast: $('#toast'), context: $('#context-menu'),
  gallery: $('#gallery'), galleryImage: $('#gallery-image'),
  galleryPrev: $('#gallery-prev'), galleryNext: $('#gallery-next'),
};
const colors = ['#e9e3f4','#dfedfb','#e5efdf','#f8e7de','#f4e3ec','#e1efed'];
const collator = new Intl.Collator('zh-CN', {numeric:true, sensitivity:'base'});
let token = '', roles = [], view = null, pending = [], galleryImages = [], galleryIndex = 0;
let activeJob = null, jobTimer = null, contextRole = null, toastTimer = null;
let requestBusy = false;
const history = new Map();
const shownJobs = new Set();

async function api(path, options = {}, allowTokenRefresh = true) {
  const headers = {...options.headers};
  if (options.method && options.method !== 'GET') headers['X-Local-Token'] = token;
  if (options.body && !(options.body instanceof FormData)) headers['Content-Type'] = 'application/json';
  const response = await fetch(path, {...options, headers});
  if (response.status === 403 && allowTokenRefresh && path !== '/api/bootstrap') {
    const bootstrap = await fetch('/api/bootstrap');
    if (bootstrap.ok) {
      token = (await bootstrap.json()).token;
      return api(path, options, false);
    }
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || '操作失败');
  return data;
}
function alertText(value) {
  els.toast.textContent = value;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {els.toast.hidden = true;}, 3500);
}
function shortName(name) {
  return /[\u3400-\u9fff]/.test(name[0]) ? name[0] : name.slice(0,2).toUpperCase();
}
function visibleRoles() {return roles.filter(role => role.trained && !role.hidden);}
async function refreshRoles() {
  roles = await api('/api/characters');
  roles.sort((a,b) => collator.compare(a.name,b.name));
  els.roles.replaceChildren();
  $('#restore-roles').hidden = !roles.some(role => role.trained && role.hidden);
  visibleRoles().forEach((role, index) => {
    const row = document.createElement('div');
    row.className = 'role-row';
    const open = document.createElement('button');
    open.className = 'side-item role-open' + (view?.name === role.name && view.mode === 'generate' ? ' active' : '');
    open.title = role.name;
    const avatar = document.createElement('span');
    avatar.className = 'role-avatar';
    avatar.style.background = colors[index % colors.length];
    avatar.textContent = shortName(role.name);
    const label = document.createElement('span');
    label.className = 'side-label role-name' + (role.name.length > 13 ? ' long' : '');
    const fullName = document.createElement('span');
    fullName.className = 'role-name-inner';
    fullName.textContent = role.name;
    label.append(fullName);
    label.onmouseenter = () => {
      const overflow = fullName.scrollWidth - label.clientWidth;
      label.style.setProperty('--scroll-distance', -Math.max(0,overflow) + 'px');
      label.classList.toggle('long',overflow > 0);
    };
    open.append(avatar,label);
    open.onclick = () => openView(role.name,'generate');
    const more = document.createElement('button');
    more.className = 'role-more';
    more.textContent = '…';
    more.title = role.name + ' 菜单';
    more.onclick = event => showContext(event, role.name);
    row.oncontextmenu = event => {event.preventDefault(); showContext(event,role.name);};
    row.append(open,more);
    els.roles.append(row);
  });
}
function home() {
  view = null;
  activeJob = null;
  clearTimeout(jobTimer);
  clearPending();
  els.cover.hidden = false;
  els.chat.hidden = true;
  els.context.hidden = true;
  refreshRoles().catch(error => alertText(error.message));
}
function clearPending() {
  pending.forEach(item => URL.revokeObjectURL(item.url));
  pending = [];
  renderPreviews();
  els.input.value = '';
}
function messageNode(role, text, urls=[]) {
  const node = document.createElement('article');
  node.className = 'message ' + role;
  if (text) {
    const paragraph = document.createElement('p');
    paragraph.textContent = text;
    node.append(paragraph);
  }
  if (urls.length) {
    const strip = document.createElement('div');
    strip.className = 'message-images';
    enableHorizontalDrag(strip);
    urls.forEach((url, index) => {
      const image = document.createElement('img');
      image.src = url; image.alt = '图片 ' + (index+1);
      image.onclick = () => openGallery(urls, index);
      strip.append(image);
    });
    node.append(strip);
  }
  return node;
}
function enableHorizontalDrag(element) {
  let pointerId = null, lastX = 0, startX = 0, dragging = false;
  element.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    pointerId = event.pointerId;
    startX = lastX = event.clientX;
    dragging = false;
  });
  element.addEventListener('pointermove', event => {
    if (pointerId === event.pointerId && (event.buttons & 1)) {
      if (!dragging && Math.abs(event.clientX - startX) < 5) return;
      if (!dragging) {
        dragging = true;
        element.setPointerCapture(event.pointerId);
      }
      element.scrollLeft += lastX - event.clientX;
      lastX = event.clientX;
    }
  });
  const endDrag = () => {pointerId = null; dragging = false;};
  element.addEventListener('pointerup', endDrag);
  element.addEventListener('pointercancel', endDrag);
}
function appendMessage(role,text,urls=[]) {
  const node = messageNode(role,text,urls);
  els.messages.append(node);
  els.messages.scrollTop = els.messages.scrollHeight;
  return node;
}
async function openView(name,mode) {
  clearTimeout(jobTimer); activeJob = null; clearPending();
  view = {name,mode,inputMode:'character'};
  const openedView = view;
  els.cover.hidden = true; els.chat.hidden = false;
  els.title.textContent = name;
  els.messages.replaceChildren();
  const key = mode + ':' + name;
  (history.get(key) || []).forEach(item => appendMessage(item.role,item.text,item.urls));
  els.tab.hidden = mode === 'generate';
  els.train.hidden = mode === 'generate';
  els.description.disabled = mode === 'generate';
  els.description.value = '';
  els.description.placeholder = mode === 'generate'
    ? '点击下方‘+’输入图片，不支持输入文字'
    : '请输入你对角色特征的描述，点击‘+’添加图片';
  els.mode.innerHTML = '角色图像输入 <span>⌄</span>';
  els.send.textContent = '↩'; els.send.title = '发送';
  els.train.textContent = '▷开始训练';
  els.input.multiple = true;
  await refreshRoles();
  try {
    const kind = mode === 'generate' ? 'generate' : 'train';
    const recent = (await api('/api/jobs/latest/' + encodeURIComponent(name)
                              + '/' + kind)).job;
    if (view !== openedView || !recent || shownJobs.has(recent.id)) return;
    if (recent.state === 'running' || recent.state === 'queued') {
      trackJob(recent.id, kind, recent.total || 1);
      if (kind === 'train') els.train.textContent = '↻重新训练';
    } else if (recent.state === 'done') {
      shownJobs.add(recent.id);
      saveMessage('system', '已完成', recent.result?.images || []);
    }
  } catch (error) {alertText(error.message);}
}
function saveMessage(role,text,urls=[]) {
  appendMessage(role,text,urls);
  if (view) {
    const key = view.mode + ':' + view.name;
    const items = history.get(key) || [];
    items.push({role,text,urls});
    history.set(key,items);
  }
}
function renderPreviews() {
  els.previews.replaceChildren();
  pending.forEach((item,index) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'preview';
    const image = document.createElement('img');
    image.src = item.url; image.alt = item.file.name;
    image.onclick = () => openGallery(pending.map(value => value.url),index);
    const remove = document.createElement('button');
    remove.textContent = '×'; remove.title = '移除图片';
    remove.onclick = () => {URL.revokeObjectURL(item.url); pending.splice(index,1); renderPreviews();};
    wrapper.append(image,remove);
    els.previews.append(wrapper);
  });
}
function openGallery(urls,index) {
  galleryImages = urls; galleryIndex = index; updateGallery();
  els.gallery.hidden = false;
}
function updateGallery() {
  els.galleryImage.src = galleryImages[galleryIndex];
  els.galleryPrev.hidden = galleryIndex === 0;
  els.galleryNext.hidden = galleryIndex === galleryImages.length - 1;
}
function closeGallery() {els.gallery.hidden = true; els.galleryImage.removeAttribute('src');}
els.galleryPrev.onclick = () => {galleryIndex--; updateGallery();};
els.galleryNext.onclick = () => {galleryIndex++; updateGallery();};
$('#gallery-close').onclick = closeGallery;
els.gallery.onclick = event => {if(event.target === els.gallery) closeGallery();};
document.addEventListener('keydown', event => {
  if (els.gallery.hidden) return;
  if (event.key === 'Escape') closeGallery();
  if (event.key === 'ArrowLeft' && galleryIndex > 0) {galleryIndex--; updateGallery();}
  if (event.key === 'ArrowRight' && galleryIndex < galleryImages.length-1) {galleryIndex++; updateGallery();}
});
enableHorizontalDrag(els.previews);
$('#pick-image').onclick = () => els.input.click();
els.input.onchange = () => {
  const files = [...els.input.files];
  if (view?.inputMode === 'mark' && (files.length !== 1 || pending.length)) {
    alertText('标志图像一次只能输入一张'); els.input.value=''; return;
  }
  pending.push(...files.map(file => ({file,url:URL.createObjectURL(file)})));
  renderPreviews();
  els.input.value = '';
};
els.mode.onclick = () => {
  els.modeMenu.hidden = !els.modeMenu.hidden;
  els.mode.setAttribute('aria-expanded', String(!els.modeMenu.hidden));
};
els.modeMenu.querySelectorAll('button').forEach(button => button.onclick = () => {
  clearPending();
  view.inputMode = button.dataset.mode;
  els.input.multiple = view.inputMode !== 'mark';
  els.mode.innerHTML = button.textContent + ' <span>⌄</span>';
  els.modeMenu.hidden = true;
  els.mode.setAttribute('aria-expanded','false');
});
async function send() {
  if (!view || requestBusy) return;
  if (activeJob) {await stopJob(); return;}
  const sentView = view;
  const description = els.description.value.trim();
  if (view.mode === 'generate') {
    if (!pending.length) {alertText('请输入图片'); return;}
    const count = pending.length;
    const urls = pending.map(item => item.url);
    const form = new FormData();
    form.append('character',view.name);
    pending.forEach(item => form.append('images',item.file));
    requestBusy = true;
    try {
      const data = await api('/api/generate',{method:'POST',body:form});
      if (view !== sentView) return;
      saveMessage('user','',urls);
      pending = []; renderPreviews();
      trackJob(data.job_id,'generate',count);
    } catch(error) {alertText(error.message);}
    finally {requestBusy = false;}
    return;
  }
  if (view.inputMode === 'mark' && (pending.length !== 1 || !description)) {
    alertText('标志图像一次只能输入一张，且必须配文字'); return;
  }
  if (!description && !pending.length) {alertText('请输入描述或添加图片'); return;}
  const form = new FormData();
  form.append('mode',view.inputMode); form.append('description',description);
  pending.forEach(item => form.append('images',item.file));
  requestBusy = true;
  try {
    const data = await api('/api/characters/'+encodeURIComponent(view.name)+'/entries',
                           {method:'POST',body:form});
    if (view !== sentView) return;
    saveMessage('user',description,data.images);
    clearPending(); els.description.value = '';
  } catch(error) {alertText(error.message);}
  finally {requestBusy = false;}
}
async function beginTraining() {
  if (!view || view.mode !== 'edit' || requestBusy) return;
  const trainingView = view;
  requestBusy = true;
  try {
    if (activeJob) await stopJob(true);
    if (view !== trainingView) return;
    const data = await api('/api/train',{method:'POST',
                                       body:JSON.stringify({name:trainingView.name})});
    if (view !== trainingView) return;
    trackJob(data.job_id,'train',0);
    els.train.textContent = '↻重新训练';
  } catch(error) {alertText(error.message);}
  finally {requestBusy = false;}
}
function trackJob(jobId,kind,total) {
  activeJob = {id:jobId,kind,total,viewName:view.name};
  els.send.textContent = '□'; els.send.title = kind === 'train' ? '暂停训练' : '停止转化';
  const panel = document.createElement('article');
  panel.className = 'message system progress-box';
  panel.innerHTML = '<div class="progress-title"></div><pre></pre>';
  els.messages.append(panel);
  activeJob.panel = panel;
  pollJob();
}
async function pollJob() {
  const tracked = activeJob;
  if (!tracked) return;
  try {
    const job = await api('/api/jobs/'+tracked.id);
    if (tracked !== activeJob) return;
    const saved = job.log_tail.filter(line => line.includes('[test] 已保存:')).length;
    const title = tracked.kind === 'generate'
      ? '图片转化中……（'+Math.min(saved+1,tracked.total)+'/'+tracked.total+'）'
      : job.phase;
    tracked.panel.querySelector('.progress-title').textContent =
      job.state === 'running' || job.state === 'queued' ? title : job.phase;
    const pre = tracked.panel.querySelector('pre');
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 12;
    pre.textContent = job.log_tail.join('\n');
    if (atBottom) pre.scrollTop = pre.scrollHeight;
    els.messages.scrollTop = els.messages.scrollHeight;
    if (job.state === 'done' || job.state === 'paused' || job.state === 'error') {
      clearTimeout(jobTimer); activeJob = null;
      els.send.textContent = '↩'; els.send.title = '发送';
      if (tracked.kind === 'train')
        els.train.textContent = job.state === 'paused' ? '▷开始训练' : '↻重新训练';
      if (job.state === 'done') {
        shownJobs.add(tracked.id);
        if (tracked.kind === 'generate') saveMessage('system','已完成',job.result?.images || []);
        else {saveMessage('system','已完成'); await refreshRoles();}
      } else if (job.state === 'paused') {
        saveMessage('system',tracked.kind === 'train'
          ? '训练暂停，点击“▷开始训练”重新开始。' : '转化已停止',job.result?.images || []);
      } else saveMessage('system','任务失败，请查看上方运行输出。');
      return;
    }
    jobTimer = setTimeout(pollJob,900);
  } catch(error) {alertText(error.message); jobTimer=setTimeout(pollJob,2500);}
}
async function stopJob(forRestart=false) {
  const tracked=activeJob;
  if (!tracked) return;
  try {
    await api('/api/jobs/'+tracked.id+'/pause',{method:'POST'});
    clearTimeout(jobTimer);
    activeJob = null;
    els.send.textContent='↩';
    if (tracked.kind==='train') els.train.textContent='▷开始训练';
    if (!forRestart) {
      await new Promise(resolve => setTimeout(resolve,250));
      const state = await api('/api/jobs/'+tracked.id);
      saveMessage('system',tracked.kind==='train'
        ? '训练暂停，点击“▷开始训练”重新开始。' : '转化已停止',
        state.result?.images || []);
    }
  } catch(error) {alertText(error.message);}
}
els.send.onclick = send;
els.train.onclick = beginTraining;
$('#open-folder').onclick = async () => {
  if (!view) return;
  try {
    await api('/api/open-folder',{method:'POST',body:JSON.stringify({
      name:view.name,kind:view.mode==='generate'?'result':'dataset'})});
  } catch(error) {alertText(error.message);}
};
$('#home').onclick = home;
const nameDialog=$('#name-dialog'), nameInput=$('#name-input');
let nameAction='add';
function askName(action) {
  nameAction=action;
  $('#name-title').textContent=action==='add'?'添加新角色':'重训角色';
  $('#name-toggle').hidden=action==='add';
  $('#name-options').hidden=true;
  nameInput.value='';
  nameDialog.showModal();
  nameInput.focus();
}
$('#add-role').onclick=()=>askName('add');
$('#retrain-role').onclick=()=>askName('retrain');
$('#name-toggle').onclick=()=>{
  const options=$('#name-options');
  options.replaceChildren();
  roles.filter(role=>role.dataset).forEach(role=>{
    const button=document.createElement('button');
    button.type='button';
    button.textContent=role.name;
    button.onclick=()=>{nameInput.value=role.name;options.hidden=true;};
    options.append(button);
  });
  options.hidden=!options.hidden;
};
$('#name-form').addEventListener('submit',async event=>{
  if (event.submitter?.value!=='confirm') return;
  event.preventDefault();
  const name=nameInput.value.trim();
  if (!name) {alertText('请输入角色名称');return;}
  try {
    if (nameAction==='add') await api('/api/characters',{
      method:'POST',body:JSON.stringify({name})});
    else if (!roles.some(role=>role.name===name && role.dataset))
      throw new Error('未找到该角色的数据集');
    nameDialog.close();
    await openView(name,'edit');
  } catch(error) {alertText(error.message);}
});
function showContext(event,name) {
  event.stopPropagation();
  contextRole=name;
  const menu=els.context;
  menu.hidden=false;
  menu.style.top=Math.min(event.clientY,innerHeight-130)+'px';
  menu.style.left=Math.min(event.clientX+8,innerWidth-172)+'px';
}
document.addEventListener('click',event=>{
  if (!els.context.contains(event.target)) els.context.hidden=true;
});
els.context.querySelectorAll('button').forEach(button=>button.onclick=async()=>{
  const action=button.dataset.action,name=contextRole;
  els.context.hidden=true;
  if(action==='hide'){
    try{await api('/api/characters/'+encodeURIComponent(name)+'/visibility',{
      method:'POST',body:JSON.stringify({hidden:true})});await refreshRoles();}
    catch(error){alertText(error.message);}
  }else if(action==='show'){
    showHiddenRoles();
  }else if(action==='delete'){
    $('#delete-message').textContent='确定删除“'+name+'”的角色文件和权重吗？删除后需要重新训练才能添加角色。';
    $('#delete-dialog').dataset.name=name;
    $('#delete-dialog').showModal();
  }
});
function showHiddenRoles(){
  const container=$('#hidden-roles');
  container.replaceChildren();
  const hidden=roles.filter(role=>role.trained && role.hidden);
  if(!hidden.length){
    const empty=document.createElement('div');empty.className='empty-hidden';
    empty.textContent='暂无未添加角色';container.append(empty);
  }
  hidden.forEach(role=>{
    const row=document.createElement('div');row.className='hidden-row';
    const label=document.createElement('span');label.textContent=role.name;
    const add=document.createElement('button');add.textContent='添加';
    add.onclick=async()=>{
      try{await api('/api/characters/'+encodeURIComponent(role.name)+'/visibility',{
        method:'POST',body:JSON.stringify({hidden:false})});
        await refreshRoles();row.remove();
        if(!container.children.length)container.textContent='暂无未添加角色';
      }catch(error){alertText(error.message);}
    };
    row.append(label,add);container.append(row);
  });
  $('#show-dialog').showModal();
}
$('#restore-roles').onclick=showHiddenRoles;
$('#close-show').onclick=()=>$('#show-dialog').close();
$('#cancel-delete').onclick=()=>$('#delete-dialog').close();
$('#confirm-delete').onclick=async()=>{
  const name=$('#delete-dialog').dataset.name;
  try{
    await api('/api/characters/'+encodeURIComponent(name),{
      method:'DELETE',body:JSON.stringify({confirm:name})});
    $('#delete-dialog').close();
    if(view?.name===name)home();
    else await refreshRoles();
  }catch(error){alertText(error.message);}
};
async function boot(){
  try{token=(await api('/api/bootstrap')).token;await refreshRoles();}
  catch(error){alertText(error.message);}
}
boot();
