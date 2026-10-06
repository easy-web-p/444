/* ระบบ AI วินิจฉัยโรคแตงโม — ฝั่งหน้าเว็บ (vanilla JS ไม่พึ่งไลบรารีภายนอก) */
'use strict';

const state = {
  health: null,
  file: null,
  diseases: null,
  groups: {},
  pesticides: null,
  fertilizers: null,
  faq: null,
  chatHistory: [],
  chatImage: null,
  diagnosisContext: '',
  lastDiagnosis: null,
  busy: false,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ---------------------------------------------------------- utilities */

function esc(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** แปลงข้อความแบบ markdown เบา ๆ (หัวข้อ ตัวหนา รายการ) ให้เป็น HTML ที่ปลอดภัย */
function renderText(raw) {
  const lines = esc(raw).split('\n');
  const out = [];
  let inList = false;
  for (const line of lines) {
    const trimmed = line.trim();
    const bullet = trimmed.match(/^[-*•]\s+(.*)$/);
    if (bullet) {
      if (!inList) { out.push('<ul class="plain">'); inList = true; }
      out.push(`<li>${inlineFmt(bullet[1])}</li>`);
      continue;
    }
    if (inList) { out.push('</ul>'); inList = false; }
    const heading = trimmed.match(/^(#{2,4})\s+(.*)$/);
    if (heading) { out.push(`<h3>${inlineFmt(heading[2])}</h3>`); continue; }
    if (/^-{3,}$/.test(trimmed)) { out.push('<hr />'); continue; }
    if (!trimmed) {
      if (out.length && out[out.length - 1] !== '') out.push('');
      continue;
    }
    out.push(`<div>${inlineFmt(trimmed)}</div>`);
  }
  if (inList) out.push('</ul>');
  return out.join('\n');
}

function inlineFmt(text) {
  return text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

function list(items, cls = 'plain') {
  if (!items || !items.length) return '';
  return `<ul class="${cls}">${items.map((i) => `<li>${esc(i)}</li>`).join('')}</ul>`;
}

function toast(message, ms = 3600) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.remove('hidden');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add('hidden'), ms);
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`โหลดข้อมูลไม่สำเร็จ (${res.status})`);
  return res.json();
}

/* ---------------------------------------------------------- tabs */

function initTabs() {
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.tab').forEach((t) => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      $$('.panel').forEach((p) => p.classList.remove('active'));
      $(`#panel-${tab.dataset.tab}`).classList.add('active');
      loadTab(tab.dataset.tab);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });
}

async function loadTab(name) {
  try {
    if (name === 'diseases' && !state.diseases) {
      const data = await getJSON('/api/diseases');
      state.diseases = data.items;
      state.groups = data.groups;
      const sel = $('#disease-group');
      Object.entries(data.groups).forEach(([id, label]) => {
        const opt = document.createElement('option');
        opt.value = id; opt.textContent = label;
        sel.appendChild(opt);
      });
      renderDiseaseList();
    }
    if (name === 'pesticides' && !state.pesticides) {
      const data = await getJSON('/api/pesticides');
      state.pesticides = data.items;
      renderPesticideList();
    }
    if (name === 'fertilizers' && !state.fertilizers) {
      state.fertilizers = await getJSON('/api/fertilizers');
      renderFertilizers();
    }
    if (name === 'faq' && !state.faq) {
      const data = await getJSON('/api/faq');
      state.faq = data.items;
      renderFaq();
    }
  } catch (err) {
    toast(err.message || 'โหลดข้อมูลไม่สำเร็จ');
  }
}

/* ---------------------------------------------------------- health */

async function initHealth() {
  const badge = $('#status-badge');
  try {
    const health = await getJSON('/api/health');
    state.health = health;
    $('#max-mb').textContent = health.limits.max_upload_mb;
    const k = health.knowledge;
    $('#footer-stats').textContent =
      `คลังความรู้: โรคและอาการผิดปกติ ${k.diseases} รายการ · ยาและปุ๋ย ${k.products} รายการ · ` +
      `คำถามที่พบบ่อย ${k.faq} ข้อ · แหล่งอ้างอิง ${k.sources} รายการ · ปรับปรุง ${k.updated}`;
    if (health.ai_enabled) {
      badge.textContent = `พร้อมใช้งาน · ${health.model}`;
      badge.className = 'status online';
    } else {
      badge.textContent = 'โหมดออฟไลน์ · ยังไม่ได้ตั้งค่า API key';
      badge.className = 'status offline';
      $('#diagnose-result').innerHTML = `
        <div class="card"><div class="notice warn">
          <strong>ระบบทำงานในโหมดออฟไลน์</strong><br />
          ยังไม่ได้ตั้งค่า <code>ANTHROPIC_API_KEY</code> ระบบจะวิเคราะห์ภาพด้วยการวิเคราะห์สีและลักษณะแผล
          ซึ่งแยกได้เพียงกลุ่มอาการและมีความแม่นยำต่ำกว่าการใช้โมเดล AI อย่างมาก
          ส่วนการถามตอบจะใช้การค้นคืนจากคลังความรู้โดยตรง
          <br />วิธีเปิดใช้งานเต็มรูปแบบ: คัดลอก <code>.env.example</code> เป็น <code>.env</code>
          แล้วใส่ค่า <code>ANTHROPIC_API_KEY</code> จากนั้นรีสตาร์ตเซิร์ฟเวอร์
        </div></div>`;
    }
  } catch (err) {
    badge.textContent = 'เชื่อมต่อเซิร์ฟเวอร์ไม่ได้';
    badge.className = 'status offline';
  }
}

/* ---------------------------------------------------------- upload + diagnose */

function initUploader() {
  const dz = $('#dropzone');
  const input = $('#file-input');

  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  ['dragenter', 'dragover'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) setFile(file);
  });
  input.addEventListener('change', () => {
    if (input.files && input.files[0]) setFile(input.files[0]);
  });
  $('#clear-image').addEventListener('click', () => {
    state.file = null;
    input.value = '';
    $('#preview-wrap').classList.add('hidden');
    $('#dropzone').classList.remove('hidden');
    $('#analyze-btn').disabled = true;
  });
  $('#context-chips').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    const box = $('#context-input');
    const value = chip.dataset.add;
    if (box.value.includes(value)) return;
    box.value = box.value ? `${box.value.replace(/\s*$/, '')} ${value}` : value;
  });
  $('#analyze-btn').addEventListener('click', runDiagnose);
}

function setFile(file) {
  if (!file.type.startsWith('image/')) { toast('กรุณาเลือกไฟล์ภาพ'); return; }
  const limit = (state.health?.limits?.max_upload_mb || 12) * 1024 * 1024;
  if (file.size > limit) { toast(`ไฟล์ใหญ่เกิน ${limit / 1024 / 1024} MB`); return; }
  state.file = file;
  const url = URL.createObjectURL(file);
  const img = $('#preview-img');
  img.src = url;
  img.onload = () => URL.revokeObjectURL(url);
  $('#preview-wrap').classList.remove('hidden');
  $('#dropzone').classList.add('hidden');
  $('#analyze-btn').disabled = false;
}

async function runDiagnose() {
  if (!state.file || state.busy) return;
  state.busy = true;
  const btn = $('#analyze-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>กำลังวิเคราะห์ภาพ… (อาจใช้เวลา 10-40 วินาที)';
  $('#diagnose-result').innerHTML = '';

  try {
    const form = new FormData();
    form.append('image', state.file);
    form.append('context', $('#context-input').value.trim());
    const res = await fetch('/api/diagnose', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'วิเคราะห์ภาพไม่สำเร็จ');
    renderDiagnosis(data);
  } catch (err) {
    $('#diagnose-result').innerHTML =
      `<div class="card"><div class="notice danger">${esc(err.message || 'เกิดข้อผิดพลาด')}</div></div>`;
  } finally {
    state.busy = false;
    btn.disabled = false;
    btn.textContent = 'วิเคราะห์ภาพด้วย AI';
  }
}

function barClass(pct) {
  if (pct >= 60) return 'bar';
  if (pct >= 35) return 'bar low';
  return 'bar verylow';
}

function renderDiagnosis(d) {
  const parts = [];
  const engineLabel = d.engine === 'claude_vision'
    ? `วิเคราะห์ด้วยโมเดล AI (${esc(d.meta?.model || '')})`
    : 'วิเคราะห์แบบออฟไลน์จากสีและลักษณะแผล (ความแม่นยำต่ำ)';

  if (!d.is_plant_image) {
    parts.push(`<div class="card"><div class="notice warn">
      <strong>ภาพนี้อาจไม่ใช่ภาพพืชหรือส่วนของพืช</strong><br />${esc(d.summary_th)}
    </div></div>`);
  }

  // ---- สรุปผล
  const quality = d.image_quality || {};
  parts.push(`<div class="card">
    <h2>ผลวิเคราะห์</h2>
    <p class="muted small">${engineLabel}${d.engines_used?.includes('local_onnx') ? ' + โมเดลที่เทรนเอง' : ''}
      ${d.plant_part ? ` · ส่วนที่วิเคราะห์: ${esc(d.plant_part)}` : ''}
      ${d.crop_guess ? ` · พืชที่เห็น: ${esc(d.crop_guess)}` : ''}</p>
    ${d.summary_th ? `<p>${esc(d.summary_th)}</p>` : ''}
    ${quality.usable === false ? `<div class="notice warn"><strong>คุณภาพภาพยังไม่เหมาะกับการวิเคราะห์</strong>
      ${list(quality.issues)}${quality.advice?.length ? `<div class="small">คำแนะนำ:</div>${list(quality.advice)}` : ''}</div>` : ''}
    ${d.severity?.level ? `<p><span class="tag">ความรุนแรง: ${esc(d.severity.level)}</span>
      <span class="tag">พื้นที่เสียหายประมาณ ${esc(d.severity.affected_area_percent)}%</span></p>
      ${d.severity.spread_risk ? `<p class="small muted">ความเสี่ยงการลุกลาม: ${esc(d.severity.spread_risk)}</p>` : ''}` : ''}
    ${d.observations?.length ? `<div class="section-title">สิ่งที่ AI เห็นในภาพ</div>${list(d.observations)}` : ''}
  </div>`);

  // ---- ผู้ต้องสงสัย
  if (d.candidates?.length) {
    const cards = d.candidates.map((c, i) => {
      const pct = Math.max(0, Math.min(100, c.confidence));
      const tag = c.is_infectious === false
        ? '<span class="tag noninfect">ไม่ใช่โรคติดเชื้อ</span>'
        : (c.is_infectious === true ? '<span class="tag infect">โรคติดเชื้อ</span>' : '');
      return `<div class="candidate${i === 0 ? ' top' : ''}">
        <div class="candidate-head">
          <div>
            <span class="candidate-name">${esc(c.name_th)}</span>
            ${c.name_en ? `<span class="muted small"> ${esc(c.name_en)}</span>` : ''}
          </div>
          <span class="candidate-pct">${pct}%</span>
        </div>
        <div class="${barClass(pct)}"><span style="width:${pct}%"></span></div>
        <div>${tag}${c.group_th ? `<span class="tag">${esc(c.group_th)}</span>` : ''}
          ${c.in_knowledge_base ? '' : '<span class="tag warn">ไม่มีในคลังความรู้</span>'}</div>
        ${c.evidence?.length ? `<div class="small" style="margin-top:8px"><strong>วิเคราะห์จาก:</strong></div>${list(c.evidence, 'evidence')}` : ''}
        ${c.against?.length ? `<div class="small" style="margin-top:6px"><strong>ข้อที่ยังไม่ยืนยัน:</strong></div>${list(c.against, 'evidence against')}` : ''}
        ${c.disease_id ? `<button class="btn ghost small" style="margin-top:8px" data-open-disease="${esc(c.disease_id)}">ดูข้อมูลโรคนี้แบบเต็ม</button>` : ''}
      </div>`;
    }).join('');
    parts.push(`<div class="card"><h2>สาเหตุที่เป็นไปได้ และความมั่นใจ</h2>
      <p class="muted small">เปอร์เซ็นต์คือระดับความมั่นใจของการวิเคราะห์ ไม่ใช่ความรุนแรงของโรค</p>
      ${cards}</div>`);
  }

  // ---- คำเตือนเรื่องความมั่นใจ
  if (d.safety?.guidance?.length) {
    parts.push(`<div class="card"><div class="notice info">${d.safety.guidance.map((g) => `<div>• ${esc(g)}</div>`).join('')}</div></div>`);
  }

  // ---- สิ่งที่ต้องทำทันที
  if (d.immediate_actions?.length || d.need_more_checks?.length) {
    parts.push(`<div class="card">
      ${d.immediate_actions?.length ? `<h2>สิ่งที่ควรทำทันที</h2>${list(d.immediate_actions)}` : ''}
      ${d.need_more_checks?.length ? `<div class="section-title">ต้องไปตรวจเพิ่มในแปลงเพื่อยืนยัน</div>${list(d.need_more_checks)}` : ''}
    </div>`);
  }

  // ---- แผนการรักษา
  if (d.treatment) parts.push(treatmentCard(d.treatment, 'แผนการจัดการ'));
  if (d.alternative_treatment) {
    parts.push(treatmentCard(d.alternative_treatment, 'แผนสำรอง (หากเป็นสาเหตุอันดับสอง)'));
  }

  // ---- ความปลอดภัย
  const safety = d.safety || {};
  parts.push(`<div class="card">
    <h2>ความปลอดภัยในการใช้สาร</h2>
    ${safety.max_phi_days != null ? `<div class="notice warn">
      ระยะเก็บเกี่ยวปลอดภัยที่นานที่สุดของสารในแผนนี้คือ <strong>${esc(safety.max_phi_days)} วัน</strong>
      ต้องเว้นจากการพ่นครั้งสุดท้ายถึงวันเก็บเกี่ยวไม่น้อยกว่านี้</div>` : ''}
    ${safety.bee_risky_products?.length ? `<div class="notice danger">
      สารที่เป็นพิษต่อผึ้งสูงในแผนนี้: ${esc(safety.bee_risky_products.join(', '))}
      — ห้ามพ่นในช่วงดอกบาน เพราะแตงโมต้องพึ่งผึ้งผสมเกสร</div>` : ''}
    ${list(safety.rules)}
    ${safety.banned_note ? `<p class="small muted">${esc(safety.banned_note)}</p>` : ''}
  </div>`);

  const topName = (d.candidates || [])[0]?.name_th || 'ผลวิเคราะห์ล่าสุด';
  parts.push(`<div class="card">
    <button class="btn primary block" data-ask-in-chat>ถามต่อเกี่ยวกับผลนี้ในแชท</button>
    <p class="small muted" style="margin-top:10px">เช่น ถามว่าถ้าไม่มีสารที่แนะนำจะใช้อะไรแทน ต้องพ่นกี่ครั้ง หรือพ่นร่วมกับปุ๋ยได้ไหม</p>
  </div>`);

  parts.push(`<div class="card"><p class="small muted">${esc(d.disclaimer)}</p>
    ${d.meta?.image ? `<p class="small muted">ภาพต้นฉบับ ${esc(d.meta.image.original_size)} ส่งวิเคราะห์ที่ ${esc(d.meta.image.sent_size)}</p>` : ''}
    ${d.meta?.llm_error ? `<p class="small muted">หมายเหตุระบบ: ${esc(d.meta.llm_error)}</p>` : ''}
  </div>`);

  const host = $('#diagnose-result');
  host.innerHTML = parts.join('');
  host.querySelectorAll('[data-open-disease]').forEach((btn) => {
    btn.addEventListener('click', () => openDiseaseDetail(btn.dataset.openDisease, true));
  });
  host.querySelector('[data-ask-in-chat]')?.addEventListener('click', () => {
    setDiagnosisContext(summarizeDiagnosis(d), d, topName);
    $('.tab[data-tab="chat"]').click();
    $('#chat-text').focus();
    toast('พร้อมถามต่อเกี่ยวกับผลวิเคราะห์นี้แล้ว');
  });
  host.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function chemicalTable(rows, title) {
  if (!rows || !rows.length) return '';
  const body = rows.map((r) => `<tr>
    <td><strong>${esc(r.name_th)}</strong>${r.name_en ? `<br /><span class="small muted">${esc(r.name_en)}</span>` : ''}
      ${r.trade_examples?.length ? `<br /><span class="small muted">ตัวอย่างชื่อการค้า: ${esc(r.trade_examples.join(', '))}</span>` : ''}</td>
    <td>${esc(r.rate)}<br /><span class="small muted">${esc(r.mode || '')}</span></td>
    <td>${esc(r.interval || '-')}</td>
    <td>${r.phi_days != null ? esc(r.phi_days) + ' วัน' : '-'}</td>
    <td class="small">${esc(r.group || '')}${r.resistance_risk ? `<br />ดื้อยา: ${esc(r.resistance_risk)}` : ''}
      ${r.bee_toxicity ? `<br />พิษต่อผึ้ง: ${esc(r.bee_toxicity)}` : ''}</td>
    <td class="small">${esc(r.note || '')}${r.mix_cautions?.length ? `<br /><em>ข้อควรระวัง: ${esc(r.mix_cautions.join(' / '))}</em>` : ''}</td>
  </tr>`).join('');
  return `<div class="section-title">${esc(title)}</div>
    <div class="table-wrap"><table>
      <thead><tr><th>สาร</th><th>อัตราต่อน้ำ 20 ลิตร</th><th>ทุก</th><th>PHI</th><th>กลุ่ม/ความเสี่ยง</th><th>หมายเหตุ</th></tr></thead>
      <tbody>${body}</tbody></table></div>`;
}

function treatmentCard(p, title) {
  return `<div class="card">
    <h2>${esc(title)}: ${esc(p.name_th)}</h2>
    <p class="muted small">
      ${p.name_en ? esc(p.name_en) + ' · ' : ''}${p.pathogen ? 'สาเหตุ: ' + esc(p.pathogen) : ''}
      ${p.severity ? ' · ความรุนแรง: ' + esc(p.severity) : ''}
      ${p.spread ? ' · การแพร่: ' + esc(p.spread) : ''}
    </p>
    ${p.yield_loss ? `<p class="small">ความเสียหายที่อาจเกิด: ${esc(p.yield_loss)}</p>` : ''}
    ${!p.is_infectious ? `<div class="notice warn">อาการนี้ไม่ใช่โรคติดเชื้อ การพ่นสารกำจัดเชื้อราหรือแบคทีเรียจะไม่ช่วย
      ให้แก้ที่สาเหตุตามรายการด้านล่าง</div>` : ''}
    ${p.urgent?.length ? `<div class="section-title">ทำทันที</div>${list(p.urgent)}` : ''}
    ${chemicalTable(p.chemical, p.is_infectious ? 'สารเคมีที่ใช้ได้' : 'ปุ๋ยและสารที่ใช้แก้อาการ')}
    ${p.biological?.length ? `<div class="section-title">ชีวภัณฑ์และทางเลือกที่ปลอดภัยกว่า</div>
      <div class="table-wrap"><table><thead><tr><th>ชีวภัณฑ์</th><th>อัตรา</th><th>วิธีใช้และข้อควรระวัง</th></tr></thead>
      <tbody>${p.biological.map((b) => `<tr><td><strong>${esc(b.name_th)}</strong></td><td>${esc(b.rate)}</td>
        <td class="small">${esc(b.note || '')}${b.mix_cautions?.length ? `<br /><em>${esc(b.mix_cautions.join(' / '))}</em>` : ''}</td></tr>`).join('')}
      </tbody></table></div>` : ''}
    ${p.cultural?.length ? `<div class="section-title">การจัดการแปลงและวิธีเขตกรรม</div>${list(p.cultural)}` : ''}
    ${p.rotation_note ? `<p class="small"><strong>การหมุนเวียนพืชและสลับกลุ่มสาร:</strong> ${esc(p.rotation_note)}</p>` : ''}
    ${p.fertilizer_advice?.length ? `<div class="section-title">คำแนะนำปุ๋ยและธาตุอาหาร</div>${list(p.fertilizer_advice)}` : ''}
    ${p.prevention?.length ? `<div class="section-title">การป้องกันในรอบปลูกถัดไป</div>${list(p.prevention)}` : ''}
    ${p.organic_options?.length ? `<div class="section-title">ทางเลือกสำหรับระบบอินทรีย์</div>${list(p.organic_options)}` : ''}
    ${p.lookalikes?.length ? `<div class="section-title">โรคที่อาการคล้ายกันและจุดแยก</div>
      <ul class="plain">${p.lookalikes.map((l) => `<li><strong>${esc(l.name_th)}:</strong> ${esc(l.key_difference)}</li>`).join('')}</ul>` : ''}
    ${p.notes ? `<div class="notice info"><strong>ข้อสังเกตสำคัญ:</strong> ${esc(p.notes)}</div>` : ''}
    ${p.refs?.length ? `<p class="small muted">แหล่งอ้างอิง: ${p.refs.map((r) => esc(r.title || r.key)).join(' · ')}</p>` : ''}
  </div>`;
}

/* ---------------------------------------------------------- chat */

function initChat() {
  const form = $('#chat-form');
  const box = $('#chat-text');
  box.addEventListener('input', () => {
    box.style.height = 'auto';
    box.style.height = Math.min(box.scrollHeight, 150) + 'px';
  });
  box.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
  form.addEventListener('submit', (e) => { e.preventDefault(); sendChat(box.value.trim()); });
  $('#chat-suggest').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (chip) sendChat(chip.dataset.q);
  });

  const imageInput = $('#chat-image-input');
  $('#chat-attach').addEventListener('click', () => imageInput.click());
  imageInput.addEventListener('change', () => {
    if (imageInput.files && imageInput.files[0]) setChatImage(imageInput.files[0]);
  });
  $('#chat-attach-clear').addEventListener('click', clearChatImage);
  $('#chat-context-clear').addEventListener('click', () => setDiagnosisContext('', null));
  addMessage('bot',
    'สวัสดีครับ ผมเป็นผู้ช่วยหมอพืชแตงโม ถามได้เลยเรื่องอาการที่พบในแปลง ยาและอัตราการใช้ โปรแกรมปุ๋ย ' +
    'หรือการดูแลตามระยะการเจริญเติบโต\n\n**แนบภาพอาการได้เลย** โดยกดปุ่มรูปภาพข้างช่องพิมพ์ ผมจะวิเคราะห์ให้แล้วถามต่อเกี่ยวกับผลนั้นได้ทันที');
}

function addMessage(role, text) {
  const log = $('#chat-log');
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.innerHTML = role === 'bot' ? renderText(text) : esc(text);
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function setChatImage(file) {
  if (!file.type.startsWith('image/')) { toast('กรุณาเลือกไฟล์ภาพ'); return; }
  const limit = (state.health?.limits?.max_upload_mb || 12) * 1024 * 1024;
  if (file.size > limit) { toast(`ไฟล์ใหญ่เกิน ${limit / 1024 / 1024} MB`); return; }
  state.chatImage = file;
  const url = URL.createObjectURL(file);
  const thumb = $('#chat-attach-thumb');
  thumb.src = url;
  thumb.onload = () => URL.revokeObjectURL(url);
  $('#chat-attach-name').textContent = file.name;
  $('#chat-attach-preview').classList.remove('hidden');
  $('#chat-attach').classList.add('has-file');
  $('#chat-text').focus();
}

function clearChatImage() {
  state.chatImage = null;
  $('#chat-image-input').value = '';
  $('#chat-attach-preview').classList.add('hidden');
  $('#chat-attach').classList.remove('has-file');
}

/** เก็บสรุปผลวินิจฉัยไว้เป็นบริบทของบทสนทนา เพื่อให้ถามต่อเนื่องได้ */
function setDiagnosisContext(summary, diagnosis, label) {
  state.diagnosisContext = summary || '';
  if (diagnosis) state.lastDiagnosis = diagnosis;
  const banner = $('#chat-context');
  if (summary) {
    $('#chat-context-label').textContent = `กำลังคุยต่อเกี่ยวกับ: ${label || 'ผลวิเคราะห์ล่าสุด'}`;
    banner.classList.remove('hidden');
  } else {
    banner.classList.add('hidden');
  }
}

/** ย่อผลวินิจฉัยให้สั้นพอส่งเป็นบริบทให้โมเดลโดยไม่เปลืองโทเคน */
function summarizeDiagnosis(d) {
  const lines = [];
  if (d.plant_part) lines.push(`ส่วนที่วิเคราะห์: ${d.plant_part}`);
  if (d.severity?.level) {
    lines.push(`ความรุนแรง: ${d.severity.level} (พื้นที่เสียหายประมาณ ${d.severity.affected_area_percent}%)`);
  }
  (d.candidates || []).slice(0, 3).forEach((c, i) => {
    lines.push(`สาเหตุอันดับ ${i + 1}: ${c.name_th} (รหัส ${c.disease_id || 'ไม่ทราบ'}) ความมั่นใจ ${c.confidence}%`);
    if (c.evidence?.length) lines.push(`  หลักฐาน: ${c.evidence.slice(0, 3).join(' / ')}`);
  });
  if (d.treatment?.chemical?.length) {
    const names = d.treatment.chemical.slice(0, 4).map((x) => x.name_th).join(', ');
    lines.push(`สารที่ระบบแนะนำไว้แล้ว: ${names}`);
  }
  if (d.observations?.length) lines.push(`สิ่งที่เห็นในภาพ: ${d.observations.slice(0, 2).join(' / ')}`);
  return lines.join('\n');
}

/** การ์ดสรุปผลวินิจฉัยแบบย่อสำหรับแสดงในบับเบิลแชท */
function renderChatDiagnosis(d) {
  const parts = [];
  if (d.summary_th) parts.push(`<div>${esc(d.summary_th)}</div>`);
  (d.candidates || []).slice(0, 3).forEach((c) => {
    const pct = Math.max(0, Math.min(100, c.confidence));
    parts.push(`<div class="mini-candidate">
      <span>${esc(c.name_th)}</span>
      <span class="${barClass(pct)}"><span style="width:${pct}%"></span></span>
      <span class="pct">${pct}%</span>
    </div>`);
  });
  const top = (d.candidates || [])[0];
  if (top?.evidence?.length) {
    parts.push(`<div class="small" style="margin-top:6px"><strong>วิเคราะห์จาก:</strong></div>${list(top.evidence.slice(0, 3))}`);
  }
  if (d.treatment?.chemical?.length) {
    parts.push('<div class="small" style="margin-top:6px"><strong>สารที่ใช้ได้ (อ่านฉลากก่อนใช้เสมอ):</strong></div>');
    parts.push(list(d.treatment.chemical.slice(0, 3).map((x) =>
      `${x.name_th} อัตรา ${x.rate}${x.phi_days != null ? ` · เก็บเกี่ยวได้หลังพ่น ${x.phi_days} วัน` : ''}`)));
  }
  if (d.safety?.guidance?.length) {
    parts.push(`<div class="small muted" style="margin-top:6px">${esc(d.safety.guidance[0])}</div>`);
  }
  parts.push('<button class="btn ghost small" style="margin-top:8px" data-show-full-diagnosis>ดูแผนการจัดการแบบเต็ม</button>');
  return parts.join('');
}

async function sendChat(message) {
  if (state.busy) return;
  if (!message && !state.chatImage) return;
  state.busy = true;
  const box = $('#chat-text');
  box.value = '';
  box.style.height = 'auto';
  $('#chat-send').disabled = true;

  if (state.chatImage) {
    await sendChatImage(message);
    state.busy = false;
    $('#chat-send').disabled = false;
    return;
  }

  addMessage('user', message);
  const bubble = addMessage('bot', '');
  bubble.classList.add('typing');
  let answer = '';
  let sources = [];

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        history: state.chatHistory.slice(-10),
        diagnosis_context: state.diagnosisContext,
      }),
    });
    if (!res.ok || !res.body) throw new Error(`เชื่อมต่อไม่สำเร็จ (${res.status})`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split('\n\n');
      buffer = chunks.pop() || '';
      for (const chunk of chunks) {
        const line = chunk.split('\n').find((l) => l.startsWith('data: '));
        if (!line) continue;
        let payload;
        try { payload = JSON.parse(line.slice(6)); } catch { continue; }
        if (payload.type === 'meta') {
          sources = payload.sources || [];
        } else if (payload.type === 'delta') {
          answer += payload.text;
          bubble.innerHTML = renderText(answer);
          $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
        } else if (payload.type === 'error') {
          answer += `\n\n[${payload.message}]`;
          bubble.innerHTML = renderText(answer);
        }
      }
    }
  } catch (err) {
    answer = answer || `ขออภัย เกิดข้อผิดพลาด: ${err.message}`;
    bubble.innerHTML = renderText(answer);
  } finally {
    bubble.classList.remove('typing');
    if (sources.length) {
      const srcEl = document.createElement('div');
      srcEl.className = 'msg-sources';
      srcEl.textContent = 'อ้างอิงจากคลังความรู้: ' + sources.map((s) => s.title).join(' · ');
      bubble.appendChild(srcEl);
    }
    state.chatHistory.push({ role: 'user', content: message });
    state.chatHistory.push({ role: 'assistant', content: answer });
    state.busy = false;
    $('#chat-send').disabled = false;
    box.focus();
  }
}

/** แนบภาพในแชท: วิเคราะห์ภาพแล้วแสดงการ์ดสรุป พร้อมจำบริบทไว้ถามต่อ */
async function sendChatImage(message) {
  const file = state.chatImage;
  const url = URL.createObjectURL(file);
  const userEl = addMessage('user', '');
  userEl.innerHTML =
    `<img class="msg-thumb" src="${url}" alt="ภาพอาการที่ส่ง" />` +
    (message ? esc(message) : '<span class="small">ช่วยวิเคราะห์ภาพนี้ให้หน่อย</span>');

  const bubble = addMessage('bot', 'กำลังวิเคราะห์ภาพ…');
  bubble.classList.add('typing');
  clearChatImage();

  try {
    const form = new FormData();
    form.append('image', file);
    form.append('context', message || '');
    const res = await fetch('/api/diagnose', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'วิเคราะห์ภาพไม่สำเร็จ');

    bubble.innerHTML = renderChatDiagnosis(data);
    const top = (data.candidates || [])[0];
    setDiagnosisContext(summarizeDiagnosis(data), data, top ? top.name_th : 'ผลวิเคราะห์ล่าสุด');
    renderDiagnosis(data);           // เตรียมผลแบบเต็มไว้ในแท็บวิเคราะห์
    bubble.querySelector('[data-show-full-diagnosis]')?.addEventListener('click', () => {
      $('.tab[data-tab="diagnose"]').click();
    });
    bubble.querySelectorAll('[data-open-disease]').forEach((btn) =>
      btn.addEventListener('click', () => openDiseaseDetail(btn.dataset.openDisease, true)));

    state.chatHistory.push({ role: 'user', content: `ส่งภาพอาการมาให้วิเคราะห์ ${message || ''}`.trim() });
    state.chatHistory.push({
      role: 'assistant',
      content: data.summary_th || 'วิเคราะห์ภาพเรียบร้อย',
    });
    addMessage('bot', 'ถามต่อเกี่ยวกับผลนี้ได้เลยครับ เช่น "ถ้าไม่มียาตัวนี้ใช้อะไรแทน" หรือ "ต้องพ่นกี่ครั้ง"');
  } catch (err) {
    bubble.innerHTML = renderText(`ขออภัย วิเคราะห์ภาพไม่สำเร็จ: ${err.message}`);
  } finally {
    bubble.classList.remove('typing');
    $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
  }
}

/* ---------------------------------------------------------- disease library */

function initDiseaseFilters() {
  $('#disease-search').addEventListener('input', renderDiseaseList);
  $('#disease-group').addEventListener('change', renderDiseaseList);
}

function renderDiseaseList() {
  if (!state.diseases) return;
  const q = $('#disease-search').value.trim().toLowerCase();
  const group = $('#disease-group').value;
  const items = state.diseases.filter((d) => {
    if (group && d.group_id !== group) return false;
    if (!q) return true;
    const hay = [d.name_th, d.name_en, d.pathogen, (d.aliases_th || []).join(' '),
      (d.image_cues || []).join(' ')].join(' ').toLowerCase();
    return hay.includes(q);
  });
  $('#disease-count').textContent = `พบ ${items.length} รายการ จากทั้งหมด ${state.diseases.length} รายการ`;
  $('#disease-list').innerHTML = items.map((d) => `
    <button class="item" data-id="${esc(d.id)}">
      <span class="name">${esc(d.name_th)}</span>
      <span class="sub">${esc(d.name_en || '')}</span>
      <span class="sub">${esc(d.group_th)} · ความรุนแรง ${esc(d.severity || '-')}</span>
      <span class="sub">${esc((d.affected_parts || []).join(', '))}</span>
    </button>`).join('') || '<p class="muted">ไม่พบรายการที่ตรงกับคำค้น</p>';
  $$('#disease-list .item').forEach((btn) =>
    btn.addEventListener('click', () => openDiseaseDetail(btn.dataset.id)));
}

async function openDiseaseDetail(id, switchTab = false) {
  try {
    if (switchTab) {
      const tab = $('.tab[data-tab="diseases"]');
      tab.click();
      await loadTab('diseases');
    }
    const data = await getJSON(`/api/diseases/${encodeURIComponent(id)}`);
    const extra = `
      ${data.image_cues?.length ? `<div class="section-title">จุดสังเกตจากภาพ</div>${list(data.image_cues)}` : ''}
      ${data.not_image_cues?.length ? `<div class="section-title">สิ่งที่บ่งว่าไม่ใช่โรคนี้</div>${list(data.not_image_cues)}` : ''}
      ${data.symptoms ? Object.entries(data.symptoms).filter(([, v]) => v && v.length)
        .map(([part, items]) => `<div class="section-title">อาการที่${esc(part)}</div>${list(items)}`).join('') : ''}
      ${data.conditions ? `<div class="section-title">สภาพที่ทำให้เกิดโรค</div>
        <ul class="plain">${Object.entries(data.conditions).filter(([, v]) => v && v !== '-')
          .map(([k, v]) => `<li><strong>${esc(k)}:</strong> ${esc(v)}</li>`).join('')}</ul>` : ''}
      ${data.transmission?.length ? `<div class="section-title">การแพร่ระบาด</div>${list(data.transmission)}` : ''}
      ${data.survival && data.survival !== '-' ? `<p class="small"><strong>การอยู่รอดของเชื้อ:</strong> ${esc(data.survival)}</p>` : ''}
      ${data.stages_at_risk?.length ? `<p class="small"><strong>ระยะที่เสี่ยง:</strong> ${esc(data.stages_at_risk.join(', '))}</p>` : ''}`;
    const host = $('#disease-detail');
    host.innerHTML = `<div class="card"><h2>${esc(data.name_th)} ${data.name_en ? `<span class="muted small">${esc(data.name_en)}</span>` : ''}</h2>
        <p class="muted small">${esc(data.group_th)}${data.pathogen ? ' · ' + esc(data.pathogen) : ''}</p>
        ${extra}</div>`
      + treatmentCard(data, 'การจัดการ')
      + '<div class="card"><button class="btn primary block" data-ask-disease>ถามต่อเกี่ยวกับโรคนี้ในแชท</button></div>';
    host.querySelector('[data-ask-disease]')?.addEventListener('click', () => {
      const summary = [
        `ผู้ใช้กำลังดูข้อมูลโรค: ${data.name_th} (รหัส ${data.disease_id})`,
        data.pathogen ? `สาเหตุ: ${data.pathogen}` : '',
        data.severity ? `ความรุนแรง: ${data.severity}` : '',
        data.chemical?.length
          ? `สารที่คลังความรู้แนะนำ: ${data.chemical.slice(0, 4).map((x) => x.name_th).join(', ')}`
          : '',
      ].filter(Boolean).join('\n');
      setDiagnosisContext(summary, null, data.name_th);
      $('.tab[data-tab="chat"]').click();
      $('#chat-text').focus();
      toast(`พร้อมถามต่อเกี่ยวกับ${data.name_th}แล้ว`);
    });
    host.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    toast(err.message || 'โหลดข้อมูลโรคไม่สำเร็จ');
  }
}

/* ---------------------------------------------------------- pesticide library */

function initPesticideFilters() {
  $('#pesticide-search').addEventListener('input', renderPesticideList);
  $('#pesticide-type').addEventListener('change', renderPesticideList);
  $('#pesticide-organic').addEventListener('change', renderPesticideList);
}

const TYPE_LABEL = {
  fungicide: 'สารป้องกันกำจัดโรคพืช',
  bactericide: 'สารกำจัดแบคทีเรีย',
  insecticide: 'สารกำจัดแมลง',
  miticide: 'สารกำจัดไร',
  nematicide: 'สารกำจัดไส้เดือนฝอย',
  biological: 'ชีวภัณฑ์',
  adjuvant: 'สารเสริม/กับดัก',
};

function renderPesticideList() {
  if (!state.pesticides) return;
  const q = $('#pesticide-search').value.trim().toLowerCase();
  const type = $('#pesticide-type').value;
  const organicOnly = $('#pesticide-organic').checked;
  const items = state.pesticides.filter((p) => {
    if (type && p.type !== type) return false;
    if (organicOnly && !p.organic_ok) return false;
    if (!q) return true;
    const hay = [p.name_th, p.name_en, p.group, (p.trade_examples || []).join(' ')].join(' ').toLowerCase();
    return hay.includes(q);
  });
  $('#pesticide-count').textContent = `พบ ${items.length} รายการ จากทั้งหมด ${state.pesticides.length} รายการ`;
  $('#pesticide-list').innerHTML = items.map((p) => `
    <button class="item" data-id="${esc(p.id)}">
      <span class="name">${esc(p.name_th)}</span>
      <span class="sub">${esc(p.name_en || '')}</span>
      <span class="sub">${esc(TYPE_LABEL[p.type] || p.type)} · ${esc(p.group || '')}</span>
      <span class="sub">อัตรา ${esc(p.rate_20l || '-')}${p.phi_days != null ? ` · PHI ${esc(p.phi_days)} วัน` : ''}</span>
      ${p.organic_ok ? '<span class="tag ok">อินทรีย์ใช้ได้</span>' : ''}
    </button>`).join('') || '<p class="muted">ไม่พบรายการที่ตรงกับคำค้น</p>';
  $$('#pesticide-list .item').forEach((btn) =>
    btn.addEventListener('click', () => openPesticideDetail(btn.dataset.id)));
}

async function openPesticideDetail(id) {
  try {
    const p = await getJSON(`/api/pesticides/${encodeURIComponent(id)}`);
    const rows = [
      ['ชนิด', TYPE_LABEL[p.type] || p.type],
      ['กลุ่มสาร', p.group],
      ['ลักษณะการออกฤทธิ์', p.mode],
      ['สูตรที่พบในท้องตลาด', (p.formulations || []).join(', ')],
      ['ตัวอย่างชื่อการค้า', (p.trade_examples || []).join(', ')],
      ['อัตราต่อน้ำ 20 ลิตร', p.rate_20l],
      ['อัตราต่อไร่', p.rate_rai],
      ['ระยะเก็บเกี่ยวปลอดภัย (PHI)', p.phi_days != null ? `${p.phi_days} วัน` : ''],
      ['ระยะปลอดภัยก่อนเข้าแปลง (REI)', p.rei_hours != null ? `${p.rei_hours} ชั่วโมง` : ''],
      ['ระดับความเป็นพิษ (WHO)', p.who_class],
      ['ความเป็นพิษต่อผึ้ง', p.bee_toxicity],
      ['ความเสี่ยงการดื้อยา', p.resistance_risk],
      ['ใช้ได้ในระบบอินทรีย์', p.organic_ok ? 'ได้' : 'ไม่ได้'],
    ].filter(([, v]) => v);
    const host = $('#pesticide-detail');
    host.innerHTML = `<div class="card">
      <h2>${esc(p.name_th)} <span class="muted small">${esc(p.name_en || '')}</span></h2>
      <div class="table-wrap"><table><tbody>
        ${rows.map(([k, v]) => `<tr><th style="width:38%">${esc(k)}</th><td>${esc(v)}</td></tr>`).join('')}
      </tbody></table></div>
      ${p.target_details?.length ? `<div class="section-title">ใช้ควบคุม</div>
        <div class="chips">${p.target_details.map((t) => `<button class="chip" data-open-disease="${esc(t.id)}">${esc(t.name_th)}</button>`).join('')}</div>` : ''}
      ${(p.mix_cautions || p.cautions || []).length ? `<div class="section-title">ข้อควรระวัง</div>${list(p.mix_cautions || p.cautions)}` : ''}
      ${p.notes ? `<div class="notice info">${esc(p.notes)}</div>` : ''}
      <p class="small muted">อัตราที่แสดงเป็นแนวทาง ต้องอ่านฉลากผลิตภัณฑ์ที่ซื้อมาและใช้อัตราตามฉลาก</p>
    </div>`;
    host.querySelectorAll('[data-open-disease]').forEach((btn) =>
      btn.addEventListener('click', () => openDiseaseDetail(btn.dataset.openDisease, true)));
    host.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    toast(err.message || 'โหลดข้อมูลสารไม่สำเร็จ');
  }
}

/* ---------------------------------------------------------- fertilizers */

function renderFertilizers() {
  const f = state.fertilizers;
  if (!f) return;
  const prep = f.programs?.soil_prep;
  const stages = (f.programs?.stages || []).map((s) => `
    <div class="stage">
      <h4>${esc(s.stage)}</h4>
      <div class="dap">วันหลังย้ายปลูก: ${esc(s.dap)} · เป้าหมาย: ${esc(s.goal)}</div>
      <div class="section-title">ปุ๋ยที่ใช้</div>${list(s.fertilizer)}
      <p class="small"><strong>การให้น้ำ:</strong> ${esc(s.water)}</p>
      <div class="section-title">สิ่งที่ต้องเฝ้าระวังในระยะนี้</div>${list(s.watch_for)}
    </div>`).join('');

  const quickref = (f.deficiency_quickref || []).map((q) => `<tr>
    <td>${esc(q.symptom)}</td>
    <td><strong>${esc(q.likely)}</strong></td>
    <td>${esc(q.quick_fix)}</td>
    <td>${q.disease_id ? `<button class="chip" data-open-disease="${esc(q.disease_id)}">ดูรายละเอียด</button>` : ''}</td>
  </tr>`).join('');

  const products = (f.products || []).map((p) => `<tr>
    <td><strong>${esc(p.name_th)}</strong><br /><span class="small muted">${esc(p.name_en || '')}</span></td>
    <td class="small">${esc(p.nutrients || '')}</td>
    <td class="small">${esc(p.rate_20l || '-')}<br />${esc(p.rate_rai || '')}</td>
    <td class="small">${esc((p.use_stage || []).join(', '))}<br />${esc(p.mode || '')}</td>
    <td class="small">${esc((p.cautions || []).join(' / '))}</td>
  </tr>`).join('');

  const host = $('#fertilizer-content');
  host.innerHTML = `
    <div class="card">
      <h2>โปรแกรมปุ๋ยแตงโมตามระยะการเจริญเติบโต</h2>
      <p class="small muted">${esc(f.disclaimer || '')}</p>
      ${prep ? `<div class="stage"><h4>${esc(prep.title)}</h4>${list(prep.steps)}</div>` : ''}
      ${stages}
      ${f.programs?.fertigation_cautions?.length ? `<div class="notice warn">
        <strong>ข้อควรระวังในการให้ปุ๋ยทางระบบน้ำ</strong>${list(f.programs.fertigation_cautions)}</div>` : ''}
      ${f.foliar_rules?.length ? `<div class="section-title">กฎการพ่นปุ๋ยทางใบ</div>${list(f.foliar_rules)}` : ''}
    </div>
    <div class="card">
      <h2>ตารางวินิจฉัยอาการขาดธาตุอาหารอย่างเร็ว</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>อาการที่เห็น</th><th>น่าจะเป็น</th><th>แก้เฉพาะหน้า</th><th></th></tr></thead>
        <tbody>${quickref}</tbody></table></div>
    </div>
    <div class="card">
      <h2>รายการปุ๋ยและวัสดุปรับปรุงดิน (${(f.products || []).length} รายการ)</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>ชื่อ</th><th>ธาตุอาหาร</th><th>อัตรา</th><th>ระยะที่ใช้</th><th>ข้อควรระวัง</th></tr></thead>
        <tbody>${products}</tbody></table></div>
    </div>`;
  host.querySelectorAll('[data-open-disease]').forEach((btn) =>
    btn.addEventListener('click', () => openDiseaseDetail(btn.dataset.openDisease, true)));
}

/* ---------------------------------------------------------- faq */

function renderFaq() {
  const q = $('#faq-search').value.trim().toLowerCase();
  const items = (state.faq || []).filter((item) =>
    !q || `${item.q} ${item.a} ${(item.tags || []).join(' ')}`.toLowerCase().includes(q));
  $('#faq-list').innerHTML = items.map((item) => `
    <details class="faq-item">
      <summary>${esc(item.q)}</summary>
      <p>${esc(item.a)}</p>
      ${item.related?.length ? `<div class="chips">${item.related.map((r) =>
        `<button class="chip" data-open-disease="${esc(r)}">ดูข้อมูลโรคที่เกี่ยวข้อง</button>`).join('')}</div>` : ''}
    </details>`).join('') || '<p class="muted">ไม่พบคำถามที่ตรงกับคำค้น</p>';
  $$('#faq-list [data-open-disease]').forEach((btn) =>
    btn.addEventListener('click', () => openDiseaseDetail(btn.dataset.openDisease, true)));
}

/* ---------------------------------------------------------- boot */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initUploader();
  initChat();
  initDiseaseFilters();
  initPesticideFilters();
  $('#faq-search').addEventListener('input', renderFaq);
  initHealth();
});
