/* SoundGuard — app: rendering, plots, interactions */
(function(){
  'use strict';

  const state = {
    sampleId: 'dog',
    corr: { noise: 0, volume: 0, clip: 0 },
  };
  const sample = () => SG.samples.find(s => s.id === state.sampleId);

  // ---------- backend wiring ----------
  let R = null;        // latest prediction from the backend
  let Rclean = null;   // clean-baseline prediction (for the corruption comparison)

  async function fetchPredict(sampleId, corr){
    const res = await fetch('/api/predict', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ sample: sampleId, corr })
    });
    if (!res.ok){
      const e = await res.json().catch(()=>({}));
      throw new Error(e.detail || ('predict failed ('+res.status+')'));
    }
    return res.json();                       // { probs, order, conf, entropy, snrDb, ..., status, cause }
  }
  async function fetchUpload(file, corr){
    const fd = new FormData();
    fd.append('file', file);
    fd.append('noise', corr.noise); fd.append('volume', corr.volume); fd.append('clip', corr.clip);
    const res = await fetch('/api/predict/upload', { method:'POST', body: fd });
    if (!res.ok){
      const e = await res.json().catch(()=>({}));
      throw new Error(e.detail || ('upload failed ('+res.status+')'));
    }
    return res.json();
  }
  // sample chips + sliders -> /api/predict (also fetch the clean baseline for the compare panel)
  // _gen guards against out-of-order responses: if the user switches selection while a request is in
  // flight, a slow reply from the superseded selection must not overwrite the newer one — otherwise the
  // UI would show a confident but STALE label, the exact failure this project is about.
  let _gen = 0;
  async function refresh(){
    const gen = ++_gen;
    try {
      const [cur, clean] = await Promise.all([
        fetchPredict(state.sampleId, state.corr),
        fetchPredict(state.sampleId, {noise:0, volume:0, clip:0}),
      ]);
      if (gen !== _gen) return;                 // a newer selection superseded this one
      R = cur; Rclean = clean; renderResult();
    } catch (err){ if (gen === _gen) showError(err.message); }
  }
  // uploaded file / mic recording -> /api/predict/upload
  const isClean = c => !c.noise && !c.volume && !c.clip;

  async function refreshUpload(file){
    const gen = ++_gen;
    try {
      // The comparison card needs an undegraded baseline. When no slider is raised the request IS
      // the baseline; when one is, a second request supplies it. Reusing the degraded result as its
      // own baseline made the card report "no change" however far the confidence had actually moved.
      const [cur, clean] = await Promise.all([
        fetchUpload(file, state.corr),
        isClean(state.corr) ? null : fetchUpload(file, {noise:0, volume:0, clip:0}),
      ]);
      if (gen !== _gen) return;
      R = cur; Rclean = clean || cur; renderResult();
    } catch (err){ if (gen === _gen) showError(err.message); }
  }
  // dashboard numbers come only from /api/dashboard — there is no baked-in fallback
  async function loadDashboard(){
    try {
      const res = await fetch('/api/dashboard');
      if (!res.ok) throw new Error('dashboard unavailable ('+res.status+')');
      Object.assign(SG, await res.json());   // SG.reliability / riskCoverage / classAcc / metrics
      buildCharts();
    } catch (err){ showChartsUnavailable(); }
  }
  function showChartsUnavailable(){
    const msg = '<div class="chart-empty">No evaluation results available. '+
      'Run <code>python eval.py</code> to produce results/dashboard.json.</div>';
    ['chart-reliability','chart-risk','metrics-table','classacc-rows'].forEach(id=>{
      const el = document.getElementById(id); if (el) el.innerHTML = msg;
    });
  }
  // escape untrusted text before it is placed into innerHTML (backend messages, filenames)
  function esc(s){
    return String(s).replace(/[&<>"']/g, c =>
      ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  function showError(msg){
    document.querySelectorAll('[data-status-card]').forEach(card=>{
      card.className = 'card status-card reject';
      card.innerHTML = `<div class="status-body"><div class="status-head">`+
        `<div class="status-icon">&#10005;</div><div><div class="status-name">Backend error</div>`+
        `<div class="status-tag">request failed</div></div></div>`+
        `<div class="status-reason">${esc(msg)}</div>`+
        `<div class="status-action"><span class="arrow">&rarr;</span><span>Is the server running at this URL? Check the Console / Network tab.</span></div></div>`;
    });
  }
  // debounce slider-driven refreshes so dragging doesn't spam the API
  let _deb = null;
  function refreshDebounced(){ clearTimeout(_deb); _deb = setTimeout(refresh, 80); }

  // What the preprocessing keeps of any recording, whatever the file itself contains:
  // features.py resamples to 16 kHz and crops to the first four seconds.
  const KEPT_SECONDS = 4.0;

  // ---------- PRNG + helpers ----------
  function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
  const clamp = (x,a,b)=> Math.max(a, Math.min(b, x));
  const lerp = (a,b,t)=> a+(b-a)*t;

  // magma-ish colormap
  const MAGMA = [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],[229,80,100],[251,135,97],[254,194,135],[252,253,191]];
  function magma(t){ t=clamp(t,0,1)*(MAGMA.length-1); const i=Math.floor(t), f=t-i; const a=MAGMA[i], b=MAGMA[Math.min(i+1,MAGMA.length-1)];
    return [lerp(a[0],b[0],f), lerp(a[1],b[1],f), lerp(a[2],b[2],f)]; }

  // ---------- waveform ----------
  function genWave(s, c){
    const N = 900, rng = mulberry32(s.seed); const arr = new Float32Array(N);
    const noiseRng = mulberry32(s.seed*7+13);
    for (let i=0;i<N;i++){
      const t = i/N; let a = 0;
      switch(s.wave){
        case 'bursts': { // sharp gaussian bursts
          const peaks = [0.12,0.34,0.52,0.71,0.88];
          for (const p of peaks){ const d=(t-p); a += Math.exp(-(d*d)/0.00035) * (0.7+0.3*rng()) * Math.sin(t*900+p*40); }
          break; }
        case 'oscillate': { // siren: AM sine, slow sweep
          const lfo = 0.55 + 0.45*Math.sin(t*Math.PI*2*2.2);
          a = lfo * Math.sin(t*Math.PI*2*60 + 6*Math.sin(t*Math.PI*2*1.5)); break; }
        case 'dense': { // rhythmic pulses
          const pulse = Math.abs(Math.sin(t*Math.PI*22)) > 0.55 ? 1 : 0.12;
          a = pulse * (0.55+0.45*rng()) * Math.sin(t*1200); break; }
        case 'sustained': { // steady block w/ texture
          const env = clamp(Math.min(t/0.05,(1-t)/0.05,1),0,1);
          a = env * (0.5 + 0.18*Math.sin(t*Math.PI*2*8)) * (0.7+0.3*Math.sin(t*700)); break; }
        case 'busy': default: { // irregular bursts
          a = (rng()*2-1)*0.45 + 0.4*Math.sin(t*Math.PI*2*4.5)*(rng()>0.3?1:0.2); break; }
      }
      arr[i] = a;
    }
    // normalize to ~1
    let mxv = 0.0001; for (let i=0;i<N;i++) mxv = Math.max(mxv, Math.abs(arr[i]));
    for (let i=0;i<N;i++) arr[i] /= mxv;
    // corruption
    const vol = 1 - 0.86*c.volume;
    const lim = 1 - 0.82*c.clip;
    for (let i=0;i<N;i++){
      let a = arr[i]*vol;
      a += (noiseRng()*2-1) * 0.6 * c.noise;          // broadband noise fills gaps
      a = clamp(a, -lim, lim);                          // clipping flattens peaks
      arr[i] = a;
    }
    return arr;
  }

  function drawWave(canvas, s, c){
    const ctx = canvas.getContext('2d'); const W = canvas.width, H = canvas.height;
    ctx.clearRect(0,0,W,H);
    // dark instrument background + baseline grid line
    ctx.fillStyle = '#070b10'; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle = 'rgba(255,255,255,0.09)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();
    const arr = genWave(s,c); const N = arr.length; const mid = H/2; const amp = H*0.46;
    const stat = (R && R.status) || 'reliable';
    // status-coded trace: reject red, uncertain amber, otherwise the phosphor signal cyan
    const col = stat==='reject' ? '#f76a5f' : stat==='uncertain' ? '#f7b93b' : '#34e0c8';
    // filled envelope
    ctx.beginPath();
    for (let i=0;i<N;i++){ const x = i/(N-1)*W; const y = mid - arr[i]*amp; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }
    for (let i=N-1;i>=0;i--){ const x = i/(N-1)*W; const y = mid + arr[i]*amp; ctx.lineTo(x,y); }
    ctx.closePath();
    ctx.globalAlpha = 0.14; ctx.fillStyle = col; ctx.fill(); ctx.globalAlpha = 1;
    // glowing outline (oscilloscope phosphor)
    ctx.beginPath();
    for (let i=0;i<N;i++){ const x = i/(N-1)*W; const y = mid - arr[i]*amp; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }
    ctx.shadowColor = col; ctx.shadowBlur = 9; ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // ---------- mel-spectrogram ----------
  // The backend returns the spectrogram the backbone was actually given, as rows of 0..255 with the
  // lowest mel band first. Draw that. drawMelStylised below is the placeholder shown before the
  // first prediction arrives, and it is labelled as an illustration in the markup because it is one.
  function drawMelReal(canvas, mel){
    const F = mel.length, Tt = mel[0].length;
    const off = drawMelReal._off || (drawMelReal._off = document.createElement('canvas'));
    off.width = Tt; off.height = F;
    const octx = off.getContext('2d'); const img = octx.createImageData(Tt, F);
    for (let f=0; f<F; f++) for (let t=0; t<Tt; t++){
      const [r,g,b] = magma(Math.pow(mel[f][t]/255, 0.8));
      const idx = ((F-1-f)*Tt + t)*4;                          // row 0 is the lowest band, drawn at the bottom
      img.data[idx]=r; img.data[idx+1]=g; img.data[idx+2]=b; img.data[idx+3]=255;
    }
    octx.putImageData(img,0,0);
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality='high';
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(off, 0,0, canvas.width, canvas.height);
  }

  // The caption has to follow the picture. The markup says "what the model sees", which is true of
  // the measured spectrogram and false of the placeholder, and a caption left unchanged over a
  // placeholder is the defect of appendix F in miniature.
  const MEL_MEASURED = 'what the model sees';
  const MEL_PLACEHOLDER = 'illustration · no spectrogram in this response';

  function labelMel(canvas, text){
    const card = canvas.closest && canvas.closest('.card');
    const hint = card && card.querySelector('.card-head .hint');
    if (hint) hint.textContent = text;
  }

  function drawMel(canvas, s, c){
    if (R && Array.isArray(R.mel) && R.mel.length){
      labelMel(canvas, MEL_MEASURED);
      return drawMelReal(canvas, R.mel);
    }
    labelMel(canvas, MEL_PLACEHOLDER);
    return drawMelStylised(canvas, s, c);
  }

  function drawMelStylised(canvas, s, c){
    const F = 80, Tt = 200; // mel bins x frames
    const off = drawMelStylised._off || (drawMelStylised._off = document.createElement('canvas'));
    off.width = Tt; off.height = F;
    const octx = off.getContext('2d'); const img = octx.createImageData(Tt, F);
    const rng = mulberry32(s.seed*3+1);
    // base energy field
    const field = new Float32Array(F*Tt);
    function add(fc, tc, fr, tr, e){ for(let f=0;f<F;f++)for(let t=0;t<Tt;t++){ const df=(f-fc)/fr, dt=(t-tc)/tr; field[f*Tt+t]+= e*Math.exp(-(df*df+dt*dt)); } }
    switch(s.mel){
      case 'bursts': { for (const p of [0.12,0.34,0.52,0.71,0.88]){ const tc=p*Tt; for(let f=0;f<F;f++){ const e=0.9*Math.exp(-Math.pow((f-F*0.45)/(F*0.5),2)); for(let t=Math.max(0,tc-4);t<Math.min(Tt,tc+4);t++) field[f*Tt+t]+= e*(1-Math.abs(t-tc)/5); } } break; }
      case 'sweep': { for(let t=0;t<Tt;t++){ const base=0.38+0.30*Math.sin(t/Tt*Math.PI*2*2.2); const fc=base*F; for(let f=0;f<F;f++){ const d=(f-fc)/(F*0.05); field[f*Tt+t]+=0.95*Math.exp(-d*d); field[f*Tt+t]+=0.4*Math.exp(-Math.pow((f-fc*2)/(F*0.06),2)); } } break; }
      case 'dense': { for(let t=0;t<Tt;t++){ const on = Math.abs(Math.sin(t/Tt*Math.PI*22))>0.55; if(!on)continue; for(let f=0;f<F;f++){ field[f*Tt+t]+= (0.55+0.4*rng())*Math.exp(-Math.pow((f-F*0.5)/(F*0.55),2)); } } break; }
      case 'bands': { for (const [fc,e] of [[0.18,0.9],[0.33,0.55],[0.5,0.35],[0.66,0.25]]){ for(let f=0;f<F;f++){ const d=(f-fc*F)/(F*0.025); const row=e*Math.exp(-d*d); for(let t=0;t<Tt;t++) field[f*Tt+t]+= row*(0.8+0.2*Math.sin(t*0.3)); } } break; }
      case 'scatter': default: { for(let k=0;k<60;k++){ add(F*(0.25+0.5*rng()), Tt*rng(), F*0.05, Tt*0.02, 0.6+0.4*rng()); } break; }
    }
    // corruption
    const nz = c.noise, vol = 1 - 0.65*c.volume, cl = c.clip;
    for (let f=0;f<F;f++) for (let t=0;t<Tt;t++){
      let val = field[f*Tt+t]*vol;
      val += nz*(0.25 + 0.55*rng());                          // noise raises floor
      if (cl>0 && f > F*0.7) val += cl*0.5*(0.5+0.5*rng());    // clipping → high-freq harmonics
      val = clamp(val,0,1);
      const [r,g,b] = magma(Math.pow(val,0.8));
      const y = F-1-f; const idx = (y*Tt+t)*4;
      img.data[idx]=r; img.data[idx+1]=g; img.data[idx+2]=b; img.data[idx+3]=255;
    }
    octx.putImageData(img,0,0);
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality='high';
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(off, 0,0, canvas.width, canvas.height);
  }

  // ---------- render prediction / confidence / status ----------
  function pct(x){ return (x*100).toFixed(x>=0.0995?0:1); }

  function renderResult(){
    const s = sample(); const r = R; if (!r) return;
    // ---- prediction (inference tab) ----
    const lead = r.order[0];
    const pred = document.getElementById('pred-body');
    let html = `<div class="top1"><span class="label">${SG.pretty(SG.classes[lead[0]])}</span><span class="pct">${pct(lead[1])}%</span></div>`;
    html += `<div class="top1-sub">Top-1 prediction · 10-class UrbanSound8K</div>`;
    html += `<div class="bars">`;
    r.order.slice(0,3).forEach((o,i)=>{
      html += `<div class="bar-row ${i===0?'lead':''}"><span class="name">${SG.pretty(SG.classes[o[0]])}</span>`+
              `<span class="bar-track"><span class="bar-fill" style="width:${(o[1]*100).toFixed(1)}%"></span></span>`+
              `<span class="bpct">${pct(o[1])}%</span></div>`;
    });
    html += `</div>`;
    pred.innerHTML = html;

    // ---- confidence (inference tab) ----
    const cb = document.getElementById('conf-body');
    cb.innerHTML = confHTML(r);
    positionMarker('conf-body', r.conf);

    // ---- status (both tabs) ----
    document.querySelectorAll('[data-status-card]').forEach(card => renderStatus(card, r));

    // ---- corruption tab mirrors ----
    drawWave(document.getElementById('wave-cor'), s, state.corr);
    drawMel(document.getElementById('mel-cor'), s, state.corr);
    renderQMeters(r);
    renderCompare(r);

    // ---- inference plots ----
    drawWave(document.getElementById('wave-inf'), s, state.corr);
    drawMel(document.getElementById('mel-inf'), s, state.corr);
  }

  function confHTML(r){
    return `<div class="conf-top"><span class="conf-label">Confidence (max-softmax)</span></div>`+
      `<div class="conf-num ${r.status}">${pct(r.conf)}<span class="unit">%</span></div>`+
      `<div class="conf-meta">normalized entropy ${r.entropy.toFixed(2)} · 10 classes</div>`+
      `<div class="gauge"><div class="zone z-red"></div><div class="zone z-amber"></div><div class="zone z-green"></div><div class="marker"></div></div>`+
      `<div class="gauge-zones"><span>Reject</span><span>Uncertain</span><span>Reliable</span></div>`+
      `<div class="gauge-scale"><span>0</span><span>45</span><span>70</span><span>100%</span></div>`;
  }
  function positionMarker(containerId, conf){
    const m = document.querySelector(`#${containerId} .marker`); if (m) m.style.left = (conf*100)+'%';
  }

  function renderStatus(card, r){
    const meta = SG.statusMeta[r.status]; const txt = SG.reasonText(r);
    const tag = SG.causeTag[r.cause] || SG.causeTag.clean;
    card.className = 'card status-card '+r.status;
    card.innerHTML = `<div class="status-body">`+
      `<div class="status-head"><div class="status-icon">${meta.icon}</div>`+
      `<div><div class="status-name">${meta.name}</div><div class="status-tag">${tag}</div></div></div>`+
      `<div class="status-reason">${txt.reason}</div>`+
      `<div class="status-action"><span class="arrow">→</span><span><b>Suggested action:</b> ${txt.action}</span></div>`+
      `</div>`;
  }

  // Read the cut-offs from the shared table so a meter can never contradict the verdict beside it.
  function qclass(metric){
    const t = SG.gateThresholds[metric];
    if (metric==='clip') return v=> v>t.bad?'bad':v>t.warn?'warn':'ok';
    return v=> v<t.bad?'bad':v<t.warn?'warn':'ok';
  }
  function renderQMeters(r){
    const el = document.getElementById('qmeters'); if(!el) return;
    const items = [
      { k:'SNR', v:r.snrDb.toFixed(0)+' dB',  cls:qclass('snr')(r.snrDb) },
      { k:'Peak level', v:r.levelDb.toFixed(0)+' dBFS', cls:qclass('lvl')(r.levelDb) },
      { k:'Clipped', v:r.clipPct.toFixed(0)+' %',  cls:qclass('clip')(r.clipPct) },
      { k:'Entropy', v:r.entropy.toFixed(2), cls:r.entropy>0.55?'bad':r.entropy>0.3?'warn':'ok' },
    ];
    el.innerHTML = items.map(i=>`<div class="qmeter ${i.cls}"><div class="qk">${i.k}</div><div class="qv">${i.v}</div></div>`).join('');
  }
  function renderCompare(r){
    const el = document.getElementById('conf-compare'); if(!el) return;
    const clean = Rclean || r;
    const drop = clean.conf - r.conf;
    el.innerHTML = `<div class="cc-cell clean"><div class="k">Clean</div><div class="big">${pct(clean.conf)}%</div><div class="cc-delta" style="color:var(--muted)">baseline</div></div>`+
      `<div class="cc-arrow">→</div>`+
      `<div class="cc-cell"><div class="k">With corruption</div><div class="big ${drop>0.02?'drop':''}">${pct(r.conf)}%</div>`+
      `<div class="cc-delta" style="color:${drop>0.02?'var(--red-ink)':'var(--muted)'}">${drop>0.005?'−'+pct(drop)+' pts':'no change'}</div></div>`;
  }

  // ---------- sliders ----------
  function syncSliders(){
    document.querySelectorAll('[data-slider]').forEach(inp=>{
      inp.value = Math.round(state.corr[inp.dataset.slider]*100);
      const out = document.querySelector(`[data-sval="${inp.dataset.slider}-${inp.dataset.group}"]`);
      if (out) out.textContent = inp.value+'%';
    });
  }
  function bindSliders(){
    document.querySelectorAll('[data-slider]').forEach(inp=>{
      inp.addEventListener('input', ()=>{
        state.corr[inp.dataset.slider] = inp.value/100;
        syncSliders(); refreshDebounced();
      });
    });
    document.querySelectorAll('[data-reset]').forEach(b=> b.addEventListener('click', ()=>{
      state.corr = {noise:0,volume:0,clip:0}; syncSliders(); refresh();
    }));
  }

  // ---------- recording -> WAV (client-side, so the server only ever receives WAV) ----------
  function encodeWav(mono, sampleRate){
    const n = mono.length, buf = new ArrayBuffer(44 + n * 2), dv = new DataView(buf);
    let p = 0;
    const str = s => { for (let i = 0; i < s.length; i++) dv.setUint8(p++, s.charCodeAt(i)); };
    const u32 = v => { dv.setUint32(p, v, true); p += 4; };
    const u16 = v => { dv.setUint16(p, v, true); p += 2; };
    str('RIFF'); u32(36 + n * 2); str('WAVE');
    str('fmt '); u32(16); u16(1); u16(1); u32(sampleRate); u32(sampleRate * 2); u16(2); u16(16);
    str('data'); u32(n * 2);
    for (let i = 0; i < n; i++){ const x = Math.max(-1, Math.min(1, mono[i])); dv.setInt16(p, x < 0 ? x * 0x8000 : x * 0x7FFF, true); p += 2; }
    return new Blob([buf], { type: 'audio/wav' });
  }
  async function blobToWavFile(blob, name){
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) throw new Error('WebAudio is unavailable in this browser');
    const ctx = new AC();
    try {
      const audio = await ctx.decodeAudioData(await blob.arrayBuffer());
      const ch = audio.numberOfChannels, len = audio.length, mono = new Float32Array(len);
      for (let c = 0; c < ch; c++){ const d = audio.getChannelData(c); for (let i = 0; i < len; i++) mono[i] += d[i] / ch; }
      return new File([encodeWav(mono, audio.sampleRate)], name, { type: 'audio/wav' });
    } finally { ctx.close(); }
  }

  // ---------- input bar ----------
  function bindInput(){
    document.querySelectorAll('.chip[data-sample]').forEach(ch=>{
      ch.addEventListener('click', ()=>{
        state.sampleId = ch.dataset.sample;
        document.querySelectorAll('.chip[data-sample]').forEach(c=> c.setAttribute('aria-pressed', c===ch));
        document.querySelectorAll('[data-nowplaying]').forEach(n=> n.innerHTML = `<b>${esc(sample().name)}</b> · classified as ${KEPT_SECONDS.toFixed(1)} s at 16 kHz`);
        refresh();
      });
    });
    function nowPlaying(name, extra){
      // name is an untrusted uploaded filename -> escape before innerHTML
      document.querySelectorAll('[data-nowplaying]').forEach(n=> n.innerHTML = `<b>${esc(name)}</b>${extra ? (' · ' + esc(extra)) : ''}`);
    }

    // --- real file upload -> POST /api/predict/upload ---
    const fileBtn = document.getElementById('btn-upload');
    const fileInput = document.getElementById('file-input');
    fileBtn.addEventListener('click', ()=> fileInput.click());
    fileInput.addEventListener('change', ()=>{
      const f = fileInput.files[0];
      if (!f) return;
      nowPlaying(f.name, 'uploaded');
      refreshUpload(f);
    });

    // --- real mic capture (MediaRecorder) -> POST /api/predict/upload ---
    // NOTE: getUserMedia needs a secure context — works on http://localhost and https.
    const rec = document.getElementById('btn-record');
    // The pipeline keeps only the first max_length_s of a recording, so anything past the cap is
    // discarded before the model ever sees it. The cap stops the timer running away when a user
    // does not realise the button is also the stop button.
    const KEPT_S = 4, MAX_RECORD_S = 10;
    const recLabel = ()=> rec.querySelector('.lbl');
    let mediaRecorder = null, chunks = [], recTimer = null;
    rec.addEventListener('click', async ()=>{
      if (mediaRecorder && mediaRecorder.state === 'recording'){ mediaRecorder.stop(); return; }
      let stream;
      try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
      catch (e){ showError('Microphone access denied or unavailable.'); return; }
      chunks = []; mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e=> { if (e.data && e.data.size) chunks.push(e.data); };
      mediaRecorder.onstop = ()=>{
        clearInterval(recTimer);
        rec.classList.remove('recording'); recLabel().textContent = 'Record';
        rec.setAttribute('aria-label', 'Record from the microphone');
        stream.getTracks().forEach(t=> t.stop());
        // Browsers record WebM/Opus (or OGG), which the server's decoder does not read. Decode it here
        // with WebAudio and re-encode to a 16-bit PCM WAV so the upload always sends a format the
        // backend accepts — no server-side transcoder needed, works offline.
        const blob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        nowPlaying('mic_recording.wav', `recorded · first ${KEPT_S} s classified`);
        blobToWavFile(blob, 'mic_recording.wav')
          .then(file => refreshUpload(file))
          .catch(err => showError('Could not process the recording: ' + (err && err.message || err)));
      };
      rec.classList.add('recording');
      // The label says what the next click does, not what the button is doing. A label that only
      // counted upwards left no way to discover that the same button ends the recording.
      let t = 0; recLabel().textContent = 'Stop · 0.0s';
      rec.setAttribute('aria-label', 'Stop recording');
      recTimer = setInterval(()=>{
        t += 0.1;
        recLabel().textContent = `Stop · ${t.toFixed(1)}s`;
        if (t >= MAX_RECORD_S && mediaRecorder && mediaRecorder.state === 'recording'){
          mediaRecorder.stop();
        }
      }, 100);
      mediaRecorder.start();
    });
  }

  // ---------- tabs ----------
  function bindTabs(){
    document.querySelectorAll('.tab').forEach(t=>{
      t.addEventListener('click', ()=>{
        document.querySelectorAll('.tab').forEach(x=> x.setAttribute('aria-selected', x===t));
        document.querySelectorAll('.tab-panel').forEach(p=> p.classList.toggle('active', p.id==='tab-'+t.dataset.tab));
        renderResult(); // re-render canvases for the now-visible tab
      });
    });
  }

  // ---------- dashboard charts (SVG) ----------
  function buildCharts(){
    document.getElementById('chart-reliability').innerHTML = reliabilitySVG();
    document.getElementById('chart-risk').innerHTML = riskCoverageSVG();
    document.getElementById('metrics-table').innerHTML = metricsTable();
    document.getElementById('classacc-rows').innerHTML = SG.classAcc.map(c=>{
      const hasCorr = c.corrupt != null;
      return `<div class="ca-row"><div class="ca-name">${SG.pretty(c.k)}</div>`+
      `<div class="ca-track"><div class="ca-fill clean" style="width:${c.clean}%"></div>`+
      (hasCorr ? `<div class="ca-fill corrupt" style="width:${c.corrupt}%"></div>` : ``)+
      `</div>`+
      `<div class="ca-vals"><span class="cv-clean">${c.clean}%</span>`+
      (hasCorr ? `<span class="cv-corr">${c.corrupt}%</span>` : `<span class="cv-corr">—</span>`)+
      `</div></div>`;
    }).join('');
  }

  function reliabilitySVG(){
    const W=480,H=360, P=46; const x0=P,y0=H-P, x1=W-18, y1=18; const sx=v=>x0+v*(x1-x0), sy=v=>y0-v*(y0-y1);
    const d = SG.reliability; const bw=(x1-x0)/d.bins.length*0.62;
    let g='';
    // grid
    for(let i=0;i<=5;i++){ const yy=sy(i/5); g+=`<line x1="${x0}" y1="${yy}" x2="${x1}" y2="${yy}" stroke="rgba(255,255,255,0.07)"/>`;
      g+=`<text x="${x0-8}" y="${yy+4}" text-anchor="end" class="ax">${(i/5).toFixed(1)}</text>`; }
    for(let i=0;i<=5;i++){ const xx=sx(i/5); g+=`<text x="${xx}" y="${y0+18}" text-anchor="middle" class="ax">${(i/5).toFixed(1)}</text>`; }
    // ideal diagonal
    g+=`<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="rgba(255,255,255,0.30)" stroke-width="1.5" stroke-dasharray="5 4"/>`;
    // bars (clean = brand, corrupted = red), grouped
    const hasCorr = Array.isArray(d.corrupted);
    d.bins.forEach((b,i)=>{ const cx=sx(b-0.05); 
      g+=`<rect x="${cx-bw}" y="${sy(d.clean[i])}" width="${bw}" height="${y0-sy(d.clean[i])}" rx="2" fill="#34e0c8" opacity="0.92"/>`;
      if (hasCorr) g+=`<rect x="${cx}" y="${sy(d.corrupted[i])}" width="${bw}" height="${y0-sy(d.corrupted[i])}" rx="2" fill="#f76a5f" opacity="0.85"/>`;
    });
    g+=`<text x="${(x0+x1)/2}" y="${H-6}" text-anchor="middle" class="axl">confidence</text>`;
    g+=`<text transform="translate(13 ${(y0+y1)/2}) rotate(-90)" text-anchor="middle" class="axl">accuracy</text>`;
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Reliability diagram: confidence versus empirical accuracy, clean and corrupted, against the ideal diagonal"><title>Reliability diagram (calibration)</title><style>.ax{font:500 11px var(--mono,monospace);fill:#8397a8}.axl{font:600 12px 'Space Grotesk';fill:#b7c6d6}</style>${g}</svg>`;
  }

  function riskCoverageSVG(){
    const W=480,H=360,P=46; const x0=P,y0=H-P,x1=W-18,y1=18; const sx=v=>x0+v*(x1-x0), sy=v=>y0-v*(y0-y1);
    const d=SG.riskCoverage; let g='';
    for(let i=0;i<=5;i++){ const yy=sy(i/5*0.4/0.4); const val=(i/5*0.4); const yyy=sy(i/5);
      g+=`<line x1="${x0}" y1="${yyy}" x2="${x1}" y2="${yyy}" stroke="rgba(255,255,255,0.07)"/>`;
      g+=`<text x="${x0-8}" y="${yyy+4}" text-anchor="end" class="ax">${(i/5*0.4).toFixed(2)}</text>`; }
    for(let i=0;i<=5;i++){ const xx=sx(i/5); g+=`<text x="${xx}" y="${y0+18}" text-anchor="middle" class="ax">${(i/5).toFixed(1)}</text>`; }
    const hasCorr = Array.isArray(d.corrupted);
    const path=(arr)=> arr.map((v,i)=>`${i?'L':'M'}${sx(d.coverage[i]).toFixed(1)} ${sy(v/0.4).toFixed(1)}`).join(' ');
    g+=`<path d="${path(d.clean)}" fill="none" stroke="#34e0c8" stroke-width="2.5"/>`;
    if (hasCorr) g+=`<path d="${path(d.corrupted)}" fill="none" stroke="#f76a5f" stroke-width="2.5"/>`;
    d.coverage.forEach((cv,i)=>{ g+=`<circle cx="${sx(cv)}" cy="${sy(d.clean[i]/0.4)}" r="3" fill="#34e0c8"/>`;
      if (hasCorr) g+=`<circle cx="${sx(cv)}" cy="${sy(d.corrupted[i]/0.4)}" r="3" fill="#f76a5f"/>`; });
    g+=`<text x="${(x0+x1)/2}" y="${H-6}" text-anchor="middle" class="axl">coverage (fraction accepted)</text>`;
    g+=`<text transform="translate(13 ${(y0+y1)/2}) rotate(-90)" text-anchor="middle" class="axl">selective risk (error)</text>`;
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Risk-coverage curve: selective risk versus coverage, clean and corrupted"><title>Risk-coverage curve (selective prediction)</title><style>.ax{font:500 11px var(--mono,monospace);fill:#8397a8}.axl{font:600 12px 'Space Grotesk';fill:#b7c6d6}</style>${g}</svg>`;
  }

  function metricsTable(){
    const pending = !SG.reliability || SG.reliability.corrupted == null;
    let rows = SG.metrics.map(m=>{
      const fmt = v => m.pct ? (v*100).toFixed(1)+'%' : v.toFixed(3);
      const hasCorr = m.corrupt != null;
      const corrCell = hasCorr
        ? `${fmt(m.corrupt)}<span class="delta">${m.pct ? ((m.corrupt-m.clean)*100).toFixed(1)+' pts' : (m.corrupt-m.clean>0?'+':'')+(m.corrupt-m.clean).toFixed(3)}</span>`
        : `<span class="delta">pending</span>`;
      return `<tr><td class="metric-name">${m.k}<small>${m.desc}${m.lowerBetter?' · lower better':' · higher better'}</small></td>`+
        `<td class="num clean">${fmt(m.clean)}</td>`+
        `<td class="num corrupt">${corrCell}</td></tr>`;
    }).join('');
    return `<table class="metrics"><thead><tr><th>Metric</th><th>Clean</th><th>Corrupted</th></tr></thead><tbody>${rows}</tbody></table>`+
      `<div class="metric-note">15-bin ECE · frozen ResNet-18 features + MLP head · no temperature scaling (raw calibration).${pending ? ' Corrupted evaluation pending.' : ' Corrupted = mixed severity test set.'}</div>`;
  }

  // ---------- init ----------
  function init(){
    loadDashboard();
    bindTabs(); bindInput(); bindSliders();
    document.querySelectorAll('[data-nowplaying]').forEach(n=> n.innerHTML = `<b>${esc(sample().name)}</b> · classified as ${KEPT_SECONDS.toFixed(1)} s at 16 kHz`);
    syncSliders();
    refresh();
  }
  document.addEventListener('DOMContentLoaded', init);
})();