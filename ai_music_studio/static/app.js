/* Ardour AI Music Studio — Client SPA */
(() => {
  "use strict";

  const state = {
    settings: null,
    providers: [],
    skills: [],
    assets: [],
    lyrics: { title: "", raw_text: "", aligned_lines: [] },
    messages: [],
    plan: [],
    pendingAttachments: [],
    ardour: { connected: false, tool_count: 0, session: {} },
    streaming: false,
    playingAssetId: null,
  };

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  function toast(msg, ms = 2600) {
    const el = $("#toast");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => el.classList.remove("show"), ms);
  }

  function escHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderMarkdown(md) {
    let text = escHtml(md || "");
    // Fenced code blocks
    text = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, _lang, code) => `<pre><code>${code}</code></pre>`);
    // Inline code
    text = text.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    // Links
    text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    // Bold & italic
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
    // Headings
    text = text.replace(/^### (.+)$/gm, "<strong>$1</strong>");
    text = text.replace(/^## (.+)$/gm, "<strong>$1</strong>");
    // Lists
    text = text.replace(/^\s*[-*] (.+)$/gm, "<li>$1</li>");
    text = text.replace(/(<li>[\s\S]*?<\/li>)/g, "<ul>$1</ul>").replace(/<\/ul>\s*<ul>/g, "");
    // Paragraphs
    text = text.replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br/>");
    return `<p>${text}</p>`;
  }

  async function api(path, opts = {}) {
    const resp = await fetch(path, opts);
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(txt || `HTTP ${resp.status}`);
    }
    return resp.json();
  }

  // ---------- Theme Sync (Ardour XML .colors) ----------
  function applyThemeVars(themeObj) {
    const vars = (themeObj && themeObj.css_vars) || {};
    const root = document.documentElement;
    for (const [k, v] of Object.entries(vars)) {
      root.style.setProperty(k, v);
    }
  }

  async function loadThemes() {
    const data = await api("/api/themes");
    const sel = $("#themeSelect");
    sel.innerHTML = data.themes
      .map((t) => `<option value="${escHtml(t.id)}"${t.id === data.current ? " selected" : ""}>Тема: ${escHtml(t.name)}</option>`)
      .join("");
    applyThemeVars(data.active_theme);
  }

  // ---------- Ardour Engine Status ----------
  async function refreshArdourStatus() {
    const lcd = $("#ardourLcd");
    const led = $("#ardourLed");
    const txt = $("#ardourLcdText");
    try {
      const st = await api("/api/ardour/status");
      state.ardour = st;
      if (st.connected) {
        lcd.classList.remove("disconnected");
        led.className = "led green";
        const s = st.session || {};
        const bpm = s.tempoBpm ? `${Number(s.tempoBpm).toFixed(1)} BPM` : "";
        const sname = s.name || "Session";
        txt.textContent = `MCP OK | ${sname}${bpm ? " | " + bpm : ""} | ${st.tool_count} tools`;
      } else {
        lcd.classList.add("disconnected");
        led.className = "led orange";
        txt.textContent = "Ardour MCP: не подключён (:4820)";
      }
    } catch {
      lcd.classList.add("disconnected");
      led.className = "led red";
      txt.textContent = "Ardour MCP: офлайн";
    }
  }

  // ---------- Audio Player & Waveform Canvas ----------
  const player = $("#globalAudioPlayer");
  player.addEventListener("ended", () => {
    state.playingAssetId = null;
    $$(".btn-play-asset").forEach((b) => (b.textContent = "▶"));
  });

  function togglePlayAsset(assetId, btn) {
    if (state.playingAssetId === assetId && !player.paused) {
      player.pause();
      state.playingAssetId = null;
      btn.textContent = "▶";
      return;
    }
    $$(".btn-play-asset").forEach((b) => (b.textContent = "▶"));
    player.src = `/api/assets/${assetId}/audio`;
    player.play().catch(() => toast("Не удалось воспроизвести файл"));
    state.playingAssetId = assetId;
    btn.textContent = "■";
  }

  function drawWaveform(canvas, peaks, vocalInfo, durSec) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 380;
    const h = canvas.clientHeight || 46;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--ardour-bg-dark").trim() || "#101010";
    ctx.fillRect(0, 0, w, h);

    // Highlight VAD vocal phrases & breaths if present
    if (vocalInfo && durSec > 0) {
      ctx.fillStyle = "rgba(61, 204, 76, 0.16)";
      for (const ph of vocalInfo.phrases || []) {
        const x0 = (ph.start_sec / durSec) * w;
        const x1 = (ph.end_sec / durSec) * w;
        ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
      }
      ctx.fillStyle = "rgba(229, 139, 5, 0.35)";
      for (const br of vocalInfo.breaths || []) {
        const x0 = (br.start_sec / durSec) * w;
        const x1 = (br.end_sec / durSec) * w;
        ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
      }
    }

    // Center zero line (Ardour ghost track style)
    ctx.strokeStyle = "rgba(127, 127, 127, 0.35)";
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (!peaks || !peaks.length) return;
    const barW = w / peaks.length;
    ctx.fillStyle = vocalInfo ? "#78cd2f" : "#5da3c1";
    for (let i = 0; i < peaks.length; i++) {
      const amp = Math.max(0.03, peaks[i]) * (h * 0.45);
      ctx.fillRect(i * barW, h / 2 - amp, Math.max(1, barW - 0.6), amp * 2);
    }
  }

  // ---------- Render Assets (References / Vocals / Samples) ----------
  const ROLE_OPTIONS = {
    references: [
      ["general_style", "Общий стиль / вайб"],
      ["drums_groove", "Ударные и грув"],
      ["bass_lowend", "Бас и саб-низ"],
      ["vocal_mix", "Звучание вокала"],
      ["mastering", "Громкость / Мастеринг"],
    ],
    vocals: [
      ["lead_vocal", "Лид-вокал"],
      ["double_vocal", "Дабл-трек"],
      ["backing_vocal", "Бэк-вокал / Гармония"],
      ["adlibs", "Эдлибы / Вставки"],
    ],
    samples: [
      ["extra_sound", "Доп. звук / FX"],
      ["drum_oneshot", "Ударный ваншот"],
      ["melodic_loop", "Мелодический луп"],
      ["transition_fx", "Переход / Riser / Impact"],
    ],
  };

  function renderAssetCategory(category) {
    const items = state.assets.filter((a) => a.category === category);
    $(`#cnt-${category}`).textContent = items.length;
    const container = $(`#list-${category}`);
    if (!items.length) {
      container.innerHTML = `<div class="card-desc" style="padding:6px 2px;">Пока нет загруженных файлов.</div>`;
      return;
    }

    container.innerHTML = "";
    const roleOpts = ROLE_OPTIONS[category] || [];

    for (const a of items) {
      const an = a.analysis || {};
      const loud = an.loudness || {};
      const spec = an.spectral || {};
      const st = an.stereo || {};
      const tempo = an.tempo || {};
      const key = an.key || {};
      const voc = an.vocal || null;

      const card = document.createElement("div");
      card.className = `asset-card ${category === "vocals" ? "vocal" : category === "samples" ? "sample" : ""}`;

      const chips = [];
      if (an.duration_sec) chips.push(`<span class="chip neutral">${an.duration_sec.toFixed(1)}s</span>`);
      if (tempo.bpm) chips.push(`<span class="chip">${tempo.bpm} BPM</span>`);
      if (key.key && key.key !== "Unknown") chips.push(`<span class="chip blue">${escHtml(key.key)}${key.camelot ? " (" + key.camelot + ")" : ""}</span>`);
      if (loud.integrated_lufs !== undefined) chips.push(`<span class="chip amber">${loud.integrated_lufs} LUFS</span>`);
      if (loud.true_peak_db !== undefined) chips.push(`<span class="chip ${loud.clipping_risk ? "red" : "neutral"}">TP ${loud.true_peak_db} dB</span>`);
      if (st.width_label) chips.push(`<span class="chip neutral">${st.width_label}</span>`);
      if (an.sample_type) chips.push(`<span class="chip blue">${an.sample_type}</span>`);
      if (voc) {
        chips.push(`<span class="chip">Фраз: ${voc.phrase_count || 0}</span>`);
        if (voc.breaths && voc.breaths.length) chips.push(`<span class="chip amber">Вздохов: ${voc.breaths.length}</span>`);
        if (voc.pitch && voc.pitch.range_low_note) chips.push(`<span class="chip blue">${voc.pitch.range_low_note}–${voc.pitch.range_high_note}</span>`);
      }

      let eqHtml = "";
      if (spec.bands_relative_db) {
        const bands = [
          ["sub", "Sub"], ["bass", "Bas"], ["low_mid", "LMd"],
          ["mid", "Mid"], ["upper_mid", "UMd"], ["presence", "Prs"], ["air", "Air"],
        ];
        eqHtml = `<div class="eq-bars">${bands.map(([k, lbl]) => {
          const val = spec.bands_relative_db[k] ?? -35;
          const pct = Math.max(8, Math.min(100, Math.round(((val + 38) / 36) * 100)));
          return `<div class="eq-col"><div class="eq-meter"><div class="eq-fill" style="height:${pct}%"></div></div><span class="eq-lbl">${lbl}</span></div>`;
        }).join("")}</div>`;
      }

      const selectHtml = roleOpts
        .map(([val, lbl]) => `<option value="${val}"${a.role === val ? " selected" : ""}>${lbl}</option>`)
        .join("");

      const vocalAlignBtn = category === "vocals"
        ? `<button class="ardour-btn sm primary btn-align-vocal" title="Привязать строки из вкладки Текст песни к миллисекундам этого вокала">Синхр. с текстом (Whisper)</button>`
        : "";

      const alignBadge = a.alignment && a.alignment.aligned_lines && a.alignment.aligned_lines.length
        ? `<span class="chip">Синхр: ${a.alignment.method} (${a.alignment.aligned_lines.length} строк)</span>`
        : "";

      card.innerHTML = `
        <div class="asset-header">
          <button class="ardour-btn sm btn-play-asset" title="Прослушать">▶</button>
          <span class="asset-name" title="${escHtml(a.filename)}">${escHtml(a.filename)}</span>
          <button class="ardour-btn sm btn-reanalyze" title="Повторить DSP анализ">DSP</button>
          <button class="ardour-btn sm danger btn-del-asset" title="Удалить">✕</button>
        </div>
        <div class="asset-body">
          <div class="waveform-wrap"><canvas class="waveform-canvas"></canvas></div>
          <div class="dsp-chips">${chips.join("")}${alignBadge}</div>
          ${eqHtml}
          <div class="meta-row">
            <select class="ardour-select sel-role">${selectHtml}</select>
            <input class="ardour-input inp-notes" value="${escHtml(a.notes || "")}" placeholder="Пометка для ИИ (что взять / куда поставить)..." />
          </div>
          ${vocalAlignBtn ? `<div style="display:flex;justify-content:flex-end;">${vocalAlignBtn}</div>` : ""}
        </div>
      `;

      container.appendChild(card);

      const canvas = card.querySelector(".waveform-canvas");
      requestAnimationFrame(() => drawWaveform(canvas, an.waveform_peaks, voc, an.duration_sec || 0));

      card.querySelector(".btn-play-asset").addEventListener("click", (e) => togglePlayAsset(a.id, e.currentTarget));
      card.querySelector(".btn-del-asset").addEventListener("click", async () => {
        await api(`/api/assets/${a.id}`, { method: "DELETE" });
        await refreshAssets();
        toast("Файл удалён");
      });
      card.querySelector(".btn-reanalyze").addEventListener("click", async () => {
        toast("Запущен DSP-анализ...");
        await api(`/api/assets/${a.id}/analyze`, { method: "POST" });
        await refreshAssets();
        toast("DSP-анализ обновлён");
      });
      card.querySelector(".sel-role").addEventListener("change", async (e) => {
        await api(`/api/assets/${a.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ role: e.target.value }),
        });
        a.role = e.target.value;
      });
      card.querySelector(".inp-notes").addEventListener("change", async (e) => {
        await api(`/api/assets/${a.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ notes: e.target.value }),
        });
        a.notes = e.target.value;
        toast("Заметка сохранена");
      });
      const btnAlign = card.querySelector(".btn-align-vocal");
      if (btnAlign) {
        btnAlign.addEventListener("click", async () => {
          btnAlign.disabled = true;
          btnAlign.textContent = "Анализ Whisper...";
          try {
            const res = await api(`/api/assets/${a.id}/align`, { method: "POST" });
            await refreshAssets();
            await refreshLyrics();
            toast(`Вокал синхронизирован (${res.alignment.method})`);
          } catch (err) {
            toast("Ошибка синхронизации: " + err.message);
          } finally {
            btnAlign.disabled = false;
            btnAlign.textContent = "Синхр. с текстом (Whisper)";
          }
        });
      }
    }
  }

  async function refreshAssets() {
    state.assets = await api("/api/assets");
    renderAssetCategory("references");
    renderAssetCategory("vocals");
    renderAssetCategory("samples");
    renderContextPills();
  }

  async function uploadFiles(files, category) {
    for (const file of files) {
      toast(`Загрузка и анализ: ${file.name}...`);
      const fd = new FormData();
      fd.append("file", file);
      fd.append("category", category);
      fd.append("auto_analyze", "true");
      const saved = await api("/api/assets/upload", { method: "POST", body: fd });
      if (category === "attachments") {
        state.pendingAttachments.push(saved);
        renderPendingAttachments();
      }
    }
    await refreshAssets();
    toast("Файлы загружены и проанализированы");
  }

  // ---------- Lyrics ----------
  async function refreshLyrics() {
    state.lyrics = await api("/api/lyrics");
    $("#lyricsTitle").value = state.lyrics.title || "";
    $("#lyricsText").value = state.lyrics.raw_text || "";
    const lineCount = (state.lyrics.raw_text || "").split("\n").filter((l) => l.trim() && !l.trim().startsWith("[")).length;
    $("#cnt-lyrics").textContent = lineCount;

    const aligned = state.lyrics.aligned_lines || [];
    const wrap = $("#lyricsAlignedWrap");
    if (!aligned.length) {
      $("#lyricsAlignInfo").textContent = "Не синхронизировано";
      wrap.innerHTML = `<div class="card-desc">Загрузите вокал во вкладке «Вокал» и нажмите «Синхр. с текстом (Whisper)» — либо ИИ сделает это сам.</div>`;
    } else {
      $("#lyricsAlignInfo").textContent = `Строк с таймкодами: ${aligned.length}`;
      wrap.innerHTML = `
        <table class="aligned-table">
          <thead><tr><th>Секция</th><th>Таймкод</th><th>Строка</th></tr></thead>
          <tbody>
            ${aligned.map((r) => `
              <tr>
                <td>${escHtml(r.section || "")}</td>
                <td class="ts">${r.start_sec != null ? `${r.start_sec.toFixed(2)}s – ${r.end_sec.toFixed(2)}s` : "—"}</td>
                <td>${escHtml(r.text || "")}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }
    renderContextPills();
  }

  // ---------- Skills ----------
  async function refreshSkills() {
    state.skills = await api("/api/skills");
    const enabledCnt = state.skills.filter((s) => s.enabled).length;
    $("#cnt-skills").textContent = `${enabledCnt}/${state.skills.length}`;
    const list = $("#list-skills");
    list.innerHTML = "";
    for (const sk of state.skills) {
      const card = document.createElement("div");
      card.className = `skill-card${sk.enabled ? "" : " disabled"}`;
      card.innerHTML = `
        <div class="card-top">
          <div class="card-title">
            <span class="led ${sk.enabled ? "green" : ""}"></span>
            <span>${escHtml(sk.name)}</span>
            <span class="chip neutral">${escHtml(sk.category)}</span>
          </div>
          <div style="display:flex;gap:4px;">
            <button class="ardour-btn sm btn-toggle-sk">${sk.enabled ? "ВКЛ" : "ВЫКЛ"}</button>
            <button class="ardour-btn sm btn-edit-sk">Ред.</button>
            <button class="ardour-btn sm danger btn-del-sk">✕</button>
          </div>
        </div>
        <div class="card-desc">${escHtml(sk.description || "")}</div>
      `;
      card.querySelector(".btn-toggle-sk").addEventListener("click", async () => {
        await api(`/api/skills/${sk.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: !sk.enabled }),
        });
        await refreshSkills();
      });
      card.querySelector(".btn-edit-sk").addEventListener("click", () => {
        $("#skId").value = sk.id;
        $("#skName").value = sk.name;
        $("#skCategory").value = sk.category || "general";
        $("#skDesc").value = sk.description || "";
        $("#skContent").value = sk.content || "";
        $("#skillEditorBox").style.display = "flex";
      });
      card.querySelector(".btn-del-sk").addEventListener("click", async () => {
        await api(`/api/skills/${sk.id}`, { method: "DELETE" });
        await refreshSkills();
        toast("Скилл удалён");
      });
      list.appendChild(card);
    }
    renderContextPills();
  }

  // ---------- Providers & Settings ----------
  async function refreshProviders() {
    state.providers = await api("/api/providers");
    $("#cnt-providers").textContent = state.providers.length;

    const topSel = $("#providerSelect");
    const prevVal = topSel.value;
    topSel.innerHTML = state.providers
      .map((p) => `<option value="${escHtml(p.id)}"${(prevVal ? prevVal === p.id : p.is_default) ? " selected" : ""}>${escHtml(p.name)} — ${escHtml(p.model)}</option>`)
      .join("");

    const list = $("#list-providers");
    list.innerHTML = "";
    for (const p of state.providers) {
      const card = document.createElement("div");
      card.className = "provider-card";
      card.innerHTML = `
        <div class="card-top">
          <div class="card-title">
            <span class="led ${p.is_default ? "green" : p.api_key ? "cyan" : ""}"></span>
            <span>${escHtml(p.name)}</span>
            ${p.is_default ? '<span class="chip">Default</span>' : ""}
          </div>
          <div style="display:flex;gap:4px;">
            <button class="ardour-btn sm btn-edit-pr">Настроить</button>
            <button class="ardour-btn sm danger btn-del-pr">✕</button>
          </div>
        </div>
        <div class="card-desc">
          <code>${escHtml(p.base_url)}</code> · модель: <strong>${escHtml(p.model)}</strong> · ключ: ${p.api_key ? "задан" : "не задан"}
        </div>
      `;
      card.querySelector(".btn-edit-pr").addEventListener("click", () => {
        $("#prId").value = p.id;
        $("#prName").value = p.name;
        $("#prBaseUrl").value = p.base_url;
        $("#prApiKey").value = p.api_key || "";
        $("#prModel").value = p.model || "";
        $("#prTemp").value = p.temperature ?? 0.4;
        $("#prMaxTokens").value = p.max_tokens ?? 4096;
        $("#prSupportsTools").checked = p.supports_tools !== false;
        $("#prDefault").checked = !!p.is_default;
        $("#prTestResult").textContent = "";
        $("#providerFormBox").style.display = "flex";
      });
      card.querySelector(".btn-del-pr").addEventListener("click", async () => {
        await api(`/api/providers/${p.id}`, { method: "DELETE" });
        await refreshProviders();
        toast("Провайдер удалён");
      });
      list.appendChild(card);
    }
  }

  async function refreshSettings() {
    const s = await api("/api/settings");
    state.settings = s;
    $("#modeSelect").value = s.autonomy_mode || "autonomous";
    $("#setMcpUrl").value = s.ardour_mcp_url || "http://127.0.0.1:4820/mcp";
    $("#setMaxSteps").value = s.max_agent_steps || 35;
    $("#setSearchProvider").value = (s.web_search && s.web_search.provider) || "duckduckgo";
    $("#setSearchKey").value = (s.web_search && (s.web_search.api_key || s.web_search.base_url)) || "";
    $("#setSearchEnabled").checked = !s.web_search || s.web_search.enabled !== false;
    $("#setWhisperModel").value = s.whisper_model || "base";
    $("#setWhisperLang").value = s.whisper_language || "";
    $("#setWhisperEnabled").checked = s.whisper_enabled !== false;
    $("#setAutoSnapshot").checked = s.auto_snapshot !== false;
    renderContextPills();
  }

  function collectProviderForm() {
    return {
      id: $("#prId").value.trim() || `prov_${Date.now().toString(36)}`,
      name: $("#prName").value.trim() || "Custom OpenAI",
      base_url: $("#prBaseUrl").value.trim() || "https://api.openai.com/v1",
      api_key: $("#prApiKey").value.trim(),
      model: $("#prModel").value.trim() || "gpt-4o",
      temperature: parseFloat($("#prTemp").value) || 0.4,
      max_tokens: parseInt($("#prMaxTokens").value, 10) || 4096,
      supports_tools: $("#prSupportsTools").checked,
      supports_audio: false,
      custom_headers: {},
      is_default: $("#prDefault").checked,
    };
  }

  // ---------- Plan & Chat Rendering ----------
  function renderPlan(steps) {
    state.plan = steps || [];
    const bar = $("#planBar");
    if (!state.plan.length) {
      bar.style.display = "none";
      return;
    }
    bar.style.display = "block";
    const done = state.plan.filter((s) => s.status === "done").length;
    $("#planProgressText").textContent = `${done}/${state.plan.length}`;
    $("#planSteps").innerHTML = state.plan
      .map((s) => {
        const led = s.status === "done" ? "green" : s.status === "in_progress" ? "orange" : s.status === "error" ? "red" : "";
        return `<div class="plan-step ${s.status}" title="${escHtml(s.detail || "")}"><span class="led ${led}"></span><span>${escHtml(s.title)}</span></div>`;
      })
      .join("");
  }

  function renderPluginCardEl(card) {
    const el = document.createElement("div");
    el.className = "plugin-card";
    const stepsHtml = (card.steps || []).map((st) => `<li>${escHtml(st)}</li>`).join("");
    const statusLabel = card.status === "installed" ? "Установлен ✓" : card.status === "skipped" ? "Пропущен" : "Требуется действие";
    el.innerHTML = `
      <div class="plugin-card-head">
        <span>Плагин / Аддон: ${escHtml(card.plugin_name)} (${escHtml(card.format)} · ${escHtml(card.price_type)})</span>
        <span>${statusLabel}</span>
      </div>
      <div class="plugin-card-body">
        <div><strong>Зачем нужен:</strong> ${escHtml(card.purpose)}</div>
        ${card.download_url ? `<div><strong>Официальный сайт:</strong> <a href="${escHtml(card.download_url)}" target="_blank" rel="noopener">${escHtml(card.download_url)}</a></div>` : ""}
        <div class="plugin-path">
          <span>Папка Windows: ${escHtml(card.install_path_windows)}</span>
          <button class="ardour-btn sm btn-copy-path">Копировать</button>
        </div>
        ${stepsHtml ? `<ol style="margin-left:18px;">${stepsHtml}</ol>` : ""}
        <div class="plugin-actions">
          <button class="ardour-btn sm green btn-plug-installed"${card.status !== "pending" ? " disabled" : ""}>Я установил → Пересканировать и продолжить</button>
          <button class="ardour-btn sm btn-plug-skip"${card.status !== "pending" ? " disabled" : ""}>Продолжить встроенными плагинами</button>
        </div>
      </div>
    `;
    el.querySelector(".btn-copy-path").addEventListener("click", () => {
      navigator.clipboard.writeText(card.install_path_windows).then(() => toast("Путь скопирован"));
    });
    el.querySelector(".btn-plug-installed").addEventListener("click", async () => {
      await api(`/api/plugin-cards/${card.id}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "installed" }),
      });
      toast("Пересканирую плагины в Ardour...");
      await api("/api/ardour/rescan-plugins", { method: "POST" });
      await sendChatMessage(`Я установил плагин "${card.plugin_name}". Пересканируй доступные плагины (plugin_list_available) и продолжи продакшн с его использованием.`);
    });
    el.querySelector(".btn-plug-skip").addEventListener("click", async () => {
      await api(`/api/plugin-cards/${card.id}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "skipped" }),
      });
      await sendChatMessage(`Продолжай без "${card.plugin_name}", используй встроенные или уже установленные в Ardour плагины.`);
    });
    return el;
  }

  function createMessageEl(msg) {
    const el = document.createElement("div");
    el.className = `msg ${msg.role}`;
    const roleLabel = msg.role === "user" ? "Вы" : "AI Music Producer";
    const snapBadge = msg.snapshot_name ? `<span class="chip">Snapshot: ${escHtml(msg.snapshot_name)}</span>` : "";

    el.innerHTML = `
      <div class="msg-head">
        <strong>${roleLabel}</strong>
        <div style="display:flex;gap:6px;align-items:center;">${snapBadge}</div>
      </div>
      <div class="msg-body">${renderMarkdown(msg.content || "")}</div>
    `;

    if (msg.attachments && msg.attachments.length) {
      const attBar = document.createElement("div");
      attBar.style.cssText = "padding:0 10px 7px;display:flex;flex-wrap:wrap;gap:4px;";
      attBar.innerHTML = msg.attachments.map((a) => `<span class="chip blue">📎 ${escHtml(a.filename)}</span>`).join("");
      el.appendChild(attBar);
    }

    if (msg.plugin_cards && msg.plugin_cards.length) {
      for (const c of msg.plugin_cards) {
        el.appendChild(renderPluginCardEl(c));
      }
    }

    if (msg.tool_calls && msg.tool_calls.length) {
      const det = document.createElement("details");
      det.className = "tool-log";
      det.innerHTML = `
        <summary>Лог движка Ardour / Инструменты (${msg.tool_calls.length})</summary>
        <div class="tool-list">
          ${msg.tool_calls.map((t) => `
            <div class="tool-row">
              <span class="led ${t.ok ? "green" : "red"}"></span>
              <span class="tname">${escHtml(t.tool_name)}</span>
              <span class="tdur">${t.duration_ms}ms</span>
              <span class="tres" title="${escHtml(t.result_summary)}">${escHtml(t.result_summary)}</span>
            </div>
          `).join("")}
        </div>
      `;
      el.appendChild(det);
    }
    return el;
  }

  function isChatNearBottom() {
    const feed = $("#chatFeed");
    if (!feed) return true;
    return feed.scrollHeight - feed.scrollTop - feed.clientHeight < 70;
  }

  function updateScrollBottomBtn() {
    const feed = $("#chatFeed");
    const btn = $("#btnScrollBottom");
    if (!feed || !btn) return;
    const hasScroll = feed.scrollHeight - feed.clientHeight > 40;
    const gap = feed.scrollHeight - feed.scrollTop - feed.clientHeight;
    btn.style.display = hasScroll && gap > 50 ? "inline-flex" : "none";
  }

  function scrollChatToBottom(force = false) {
    const feed = $("#chatFeed");
    if (!feed) return;
    if (force || isChatNearBottom()) {
      feed.scrollTop = feed.scrollHeight;
    }
    updateScrollBottomBtn();
  }

  function renderChatFeed() {
    const feed = $("#chatFeed");
    feed.innerHTML = "";
    if (!state.messages.length) {
      feed.innerHTML = `
        <div class="empty-welcome">
          <h2>AI Music Studio готов к работе с движком Ardour</h2>
          <p>Загрузите материалы во вкладках слева (<strong>Референсы</strong>, <strong>Вокал</strong>, <strong>Звуки</strong>, <strong>Текст песни</strong>), подключите нужные <strong>Скиллы</strong> и любой <strong>OpenAI-compatible</strong> API-провайдер — и опишите, какой трек нужно создать.</p>
          <p>Нейросеть сама проанализирует аудио, выставит темп/тональность, напишет партии, подрежет и выровняет вокал по тексту, сведёт дорожки и мастер-шину, а при нехватке плагинов найдёт их через Web Search и объяснит, как установить.</p>
          <div class="quick-prompts">
            <button class="ardour-btn sm quick-p">Собери полный трек по моим референсам, вокалу и тексту</button>
            <button class="ardour-btn sm quick-p">Подрежь и выровняй вокал, убери паузы и сведи с ревербом</button>
            <button class="ardour-btn sm quick-p">Проверь установленные плагины и посоветуй бесплатные VST3 для моего стиля</button>
          </div>
        </div>
      `;
      feed.querySelectorAll(".quick-p").forEach((b) =>
        b.addEventListener("click", () => {
          $("#chatInput").value = b.textContent;
          $("#chatInput").focus();
        })
      );
      updateScrollBottomBtn();
      return;
    }
    for (const m of state.messages) {
      feed.appendChild(createMessageEl(m));
    }
    scrollChatToBottom(true);
    requestAnimationFrame(() => scrollChatToBottom(true));
  }

  function renderContextPills() {
    const refs = state.assets.filter((a) => a.category === "references").length;
    const vocs = state.assets.filter((a) => a.category === "vocals").length;
    const smps = state.assets.filter((a) => a.category === "samples").length;
    const sks = state.skills.filter((s) => s.enabled).length;
    const hasLyrics = !!(state.lyrics.raw_text && state.lyrics.raw_text.trim());
    const wsOn = state.settings && state.settings.web_search && state.settings.web_search.enabled;

    $("#ctxPills").innerHTML = `
      <span class="chip ${refs ? "blue" : "neutral"}">Реф: ${refs}</span>
      <span class="chip ${vocs ? "" : "neutral"}">Вокал: ${vocs}</span>
      <span class="chip ${smps ? "amber" : "neutral"}">Звуки: ${smps}</span>
      <span class="chip ${hasLyrics ? "" : "neutral"}">Текст: ${hasLyrics ? "есть" : "нет"}</span>
      <span class="chip neutral">Скиллы: ${sks}</span>
      <span class="chip ${wsOn ? "" : "neutral"}">Web Search: ${wsOn ? "ON" : "OFF"}</span>
    `;
  }

  function renderPendingAttachments() {
    const bar = $("#pendingAttachments");
    if (!state.pendingAttachments.length) {
      bar.style.display = "none";
      bar.innerHTML = "";
      return;
    }
    bar.style.display = "flex";
    bar.innerHTML = state.pendingAttachments
      .map((a, idx) => `<span class="chip blue">📎 ${escHtml(a.filename)} <a href="#" data-rm-att="${idx}" style="color:#fff;text-decoration:none;margin-left:4px;">✕</a></span>`)
      .join("");
    bar.querySelectorAll("[data-rm-att]").forEach((a) =>
      a.addEventListener("click", (e) => {
        e.preventDefault();
        state.pendingAttachments.splice(Number(a.dataset.rmAtt), 1);
        renderPendingAttachments();
      })
    );
  }

  // ---------- Streaming Chat Execution ----------
  async function sendChatMessage(overrideText = null) {
    if (state.streaming) return;
    const input = $("#chatInput");
    const text = (overrideText !== null ? overrideText : input.value).trim();
    if (!text) return;
    if (overrideText === null) input.value = "";

    const attachmentIds = state.pendingAttachments.map((a) => a.id);
    const attachedPreview = [...state.pendingAttachments];
    state.pendingAttachments = [];
    renderPendingAttachments();

    state.streaming = true;
    $("#btnSend").style.display = "none";
    $("#btnStop").style.display = "inline-flex";
    $("#agentStatusText").textContent = "Запускаю AI-продюсера...";

    const feed = $("#chatFeed");
    if (!state.messages.length) feed.innerHTML = "";

    // Optimistic user bubble
    const tempUser = { id: "tmp_u", role: "user", content: text, attachments: attachedPreview };
    feed.appendChild(createMessageEl(tempUser));

    // Live streaming assistant bubble
    const liveEl = document.createElement("div");
    liveEl.className = "msg assistant";
    liveEl.innerHTML = `
      <div class="msg-head">
        <strong>AI Music Producer</strong>
        <span class="chip amber live-badge">Работаю...</span>
      </div>
      <div class="msg-body live-body"></div>
      <div class="live-cards"></div>
      <details class="tool-log" open style="display:none;">
        <summary class="live-tool-sum">Инструменты (0)</summary>
        <div class="tool-list live-tool-list"></div>
      </details>
    `;
    feed.appendChild(liveEl);
    feed.scrollTop = feed.scrollHeight;

    let accText = "";
    let toolCount = 0;

    try {
      const resp = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          provider_id: $("#providerSelect").value || null,
          attachment_ids: attachmentIds,
          mode: $("#modeSelect").value,
        }),
      });
      if (!resp.ok) throw new Error(await resp.text());

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() || "";

        for (const chunk of parts) {
          const line = chunk.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;
          let ev;
          try {
            ev = JSON.parse(line.slice(5).trim());
          } catch {
            continue;
          }

          if (ev.type === "status") {
            $("#agentStatusText").textContent = ev.text;
          } else if (ev.type === "token") {
            const stick = isChatNearBottom();
            accText += ev.text;
            liveEl.querySelector(".live-body").innerHTML = renderMarkdown(accText);
            scrollChatToBottom(stick);
          } else if (ev.type === "plan") {
            renderPlan(ev.steps);
          } else if (ev.type === "tool_start") {
            const stick = isChatNearBottom();
            toolCount++;
            const log = liveEl.querySelector(".tool-log");
            log.style.display = "block";
            liveEl.querySelector(".live-tool-sum").textContent = `Лог движка Ardour / Инструменты (${toolCount})`;
            const row = document.createElement("div");
            row.className = "tool-row";
            row.dataset.tid = ev.id;
            row.innerHTML = `<span class="led orange"></span><span class="tname">${escHtml(ev.tool_name)}</span><span class="tres">выполняется...</span>`;
            liveEl.querySelector(".live-tool-list").appendChild(row);
            scrollChatToBottom(stick);
          } else if (ev.type === "tool_end") {
            const stick = isChatNearBottom();
            const tr = ev.trace;
            const row = liveEl.querySelector(`.tool-row[data-tid="${tr.id}"]`);
            if (row) {
              row.innerHTML = `<span class="led ${tr.ok ? "green" : "red"}"></span><span class="tname">${escHtml(tr.tool_name)}</span><span class="tdur">${tr.duration_ms}ms</span><span class="tres" title="${escHtml(tr.result_summary)}">${escHtml(tr.result_summary)}</span>`;
            }
            scrollChatToBottom(stick);
          } else if (ev.type === "plugin_card") {
            const stick = isChatNearBottom();
            liveEl.querySelector(".live-cards").appendChild(renderPluginCardEl(ev.card));
            scrollChatToBottom(stick);
          } else if (ev.type === "assets_updated") {
            refreshAssets();
          } else if (ev.type === "lyrics_updated") {
            refreshLyrics();
          } else if (ev.type === "error") {
            liveEl.querySelector(".live-body").innerHTML += `<p style="color:var(--ardour-alert-red);"><strong>Ошибка:</strong> ${escHtml(ev.message)}</p>`;
            toast("Ошибка: " + ev.message, 4500);
          } else if (ev.type === "done") {
            const st = await api("/api/chat/state");
            state.messages = st.messages;
            renderPlan(st.plan);
            renderChatFeed();
            refreshArdourStatus();
          }
        }
      }
    } catch (err) {
      liveEl.querySelector(".live-body").innerHTML += `<p style="color:var(--ardour-alert-red);"><strong>Ошибка соединения:</strong> ${escHtml(err.message)}</p>`;
    } finally {
      state.streaming = false;
      $("#btnSend").style.display = "inline-flex";
      $("#btnStop").style.display = "none";
      $("#agentStatusText").textContent = "Готов к работе";
    }
  }

  // ---------- Event Wiring ----------
  function bindEvents() {
    // Left context tabs
    $$(".page-tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$(".page-tab-btn").forEach((b) => b.classList.remove("active"));
        $$(".tab-panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        $(`#tab-${btn.dataset.tab}`).classList.add("active");
      });
    });

    // Drop zones
    $$(".drop-zone").forEach((dz) => {
      const inp = dz.querySelector("input[type='file']");
      const cat = dz.dataset.category;
      dz.addEventListener("click", () => inp.click());
      inp.addEventListener("change", () => {
        if (inp.files.length) uploadFiles(inp.files, cat);
        inp.value = "";
      });
      dz.addEventListener("dragover", (e) => {
        e.preventDefault();
        dz.classList.add("drag-over");
      });
      dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
      dz.addEventListener("drop", (e) => {
        e.preventDefault();
        dz.classList.remove("drag-over");
        if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files, cat);
      });
    });

    // Resizable splitter
    const splitter = $("#paneSplitter");
    const leftPane = $("#leftPane");
    let dragging = false;
    splitter.addEventListener("mousedown", () => (dragging = true));
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const w = Math.max(340, Math.min(680, e.clientX));
      leftPane.style.width = `${w}px`;
    });
    window.addEventListener("mouseup", () => (dragging = false));

    // Theme switch (syncs with Ardour themes & saves in settings)
    $("#themeSelect").addEventListener("change", async (e) => {
      const tid = e.target.value;
      const themeObj = await api(`/api/themes/${tid}`);
      applyThemeVars(themeObj);
      if (state.settings) {
        state.settings.theme_name = tid;
        await api("/api/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(state.settings),
        });
      }
      toast(`Тема Ardour: ${tid}`);
    });

    $("#btnRefreshArdour").addEventListener("click", refreshArdourStatus);

    $("#chatFeed").addEventListener("scroll", updateScrollBottomBtn);
    $("#btnScrollBottom").addEventListener("click", () => {
      scrollChatToBottom(true);
      $("#chatFeed").focus();
    });

    $("#btnUndo").addEventListener("click", async () => {
      const res = await api("/api/ardour/undo", { method: "POST" });
      toast(res.ok ? "Откат в Ardour выполнен (Undo)" : `Не удалось откатить: ${res.error}`);
    });

    $("#btnClearChat").addEventListener("click", async () => {
      await api("/api/chat/clear", { method: "POST" });
      state.messages = [];
      state.plan = [];
      renderPlan([]);
      renderChatFeed();
      toast("История чата очищена");
    });

    // Lyrics
    $$("[data-lyric-tag]").forEach((b) => {
      b.addEventListener("click", () => {
        const ta = $("#lyricsText");
        const tag = b.dataset.lyricTag;
        const prefix = ta.value && !ta.value.endsWith("\n") ? "\n\n" : "";
        ta.value += `${prefix}${tag}\n`;
        ta.focus();
      });
    });
    $("#btnSaveLyrics").addEventListener("click", async () => {
      await api("/api/lyrics", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: $("#lyricsTitle").value,
          raw_text: $("#lyricsText").value,
        }),
      });
      await refreshLyrics();
      toast("Текст песни сохранён");
    });

    // Skills
    $("#btnNewSkill").addEventListener("click", () => {
      $("#skId").value = "";
      $("#skName").value = "";
      $("#skDesc").value = "";
      $("#skContent").value = "";
      $("#skCategory").value = "general";
      $("#skillEditorBox").style.display = "flex";
    });
    $("#btnCancelSkill").addEventListener("click", () => ($("#skillEditorBox").style.display = "none"));
    $("#btnSaveSkill").addEventListener("click", async () => {
      const name = $("#skName").value.trim();
      if (!name) return toast("Введите название скилла");
      await api("/api/skills", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: $("#skId").value.trim() || "",
          name,
          category: $("#skCategory").value,
          description: $("#skDesc").value.trim(),
          content: $("#skContent").value,
          enabled: true,
        }),
      });
      $("#skillEditorBox").style.display = "none";
      await refreshSkills();
      toast("Скилл сохранён");
    });
    $("#btnImportSkillMd").addEventListener("click", () => $("#skillFileInput").click());
    $("#skillFileInput").addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const content = await file.text();
      await api("/api/skills", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: file.name.replace(/\.(md|txt)$/i, ""),
          name: file.name.replace(/\.(md|txt)$/i, ""),
          category: "general",
          description: "Импортированный скилл",
          content,
          enabled: true,
        }),
      });
      e.target.value = "";
      await refreshSkills();
      toast("Скилл импортирован");
    });

    // Providers
    $("#btnNewProvider").addEventListener("click", () => {
      $("#prId").value = "";
      $("#prName").value = "";
      $("#prBaseUrl").value = "https://api.openai.com/v1";
      $("#prApiKey").value = "";
      $("#prModel").value = "";
      $("#prTemp").value = "0.4";
      $("#prMaxTokens").value = "4096";
      $("#prSupportsTools").checked = true;
      $("#prDefault").checked = false;
      $("#prTestResult").textContent = "";
      $("#providerFormBox").style.display = "flex";
    });
    $("#btnCancelProvider").addEventListener("click", () => ($("#providerFormBox").style.display = "none"));
    $("#btnSaveProvider").addEventListener("click", async () => {
      const prov = collectProviderForm();
      await api("/api/providers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(prov),
      });
      $("#providerFormBox").style.display = "none";
      await refreshProviders();
      toast(`Провайдер «${prov.name}» сохранён`);
    });
    $("#btnFetchModels").addEventListener("click", async () => {
      $("#prTestResult").textContent = "Загружаю /models...";
      const res = await api("/api/providers/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectProviderForm()),
      });
      if (res.ok && res.models.length) {
        $("#prModelList").innerHTML = res.models.map((m) => `<option value="${escHtml(m)}"></option>`).join("");
        if (!$("#prModel").value) $("#prModel").value = res.models[0];
        $("#prTestResult").textContent = `Загружено моделей: ${res.models.length} (выберите из списка в поле «Модель»)`;
      } else {
        $("#prTestResult").textContent = `Не удалось получить список: ${res.error || "пустой ответ"}`;
      }
    });
    $("#btnTestProvider").addEventListener("click", async () => {
      $("#prTestResult").textContent = "Проверка соединения...";
      const res = await api("/api/providers/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectProviderForm()),
      });
      $("#prTestResult").textContent = res.message || (res.ok ? "OK" : "Ошибка");
      if (res.models && res.models.length) {
        $("#prModelList").innerHTML = res.models.map((m) => `<option value="${escHtml(m)}"></option>`).join("");
      }
    });

    // Settings
    $("#btnSaveSettings").addEventListener("click", async () => {
      const sp = $("#setSearchProvider").value;
      const sk = $("#setSearchKey").value.trim();
      const payload = {
        ardour_mcp_url: $("#setMcpUrl").value.trim() || "http://127.0.0.1:4820/mcp",
        theme_name: $("#themeSelect").value || "dark",
        autonomy_mode: $("#modeSelect").value || "autonomous",
        auto_snapshot: $("#setAutoSnapshot").checked,
        max_agent_steps: parseInt($("#setMaxSteps").value, 10) || 35,
        whisper_model: $("#setWhisperModel").value || "base",
        whisper_enabled: $("#setWhisperEnabled").checked,
        whisper_language: $("#setWhisperLang").value.trim(),
        web_search: {
          enabled: $("#setSearchEnabled").checked,
          provider: sp,
          api_key: sp === "searxng" ? "" : sk,
          base_url: sp === "searxng" ? sk : "",
          max_results: 6,
        },
      };
      await api("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      await refreshSettings();
      await refreshArdourStatus();
      toast("Настройки студии сохранены");
    });

    // Chat attachments & send/stop
    $("#btnAttachChat").addEventListener("click", () => $("#chatFileInput").click());
    $("#chatFileInput").addEventListener("change", async (e) => {
      if (e.target.files.length) await uploadFiles(e.target.files, "attachments");
      e.target.value = "";
    });
    $("#btnSend").addEventListener("click", () => sendChatMessage());
    $("#btnStop").addEventListener("click", async () => {
      await api("/api/chat/stop", { method: "POST" });
      toast("Остановка агента...");
    });
    $("#chatInput").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }

  // ---------- Bootstrap ----------
  async function init() {
    bindEvents();
    await loadThemes();
    const [, , , , , chatState] = await Promise.all([
      refreshSettings(),
      refreshProviders(),
      refreshSkills(),
      refreshAssets(),
      refreshLyrics(),
      api("/api/chat/state"),
    ]);
    state.messages = chatState.messages || [];
    renderPlan(chatState.plan || []);
    renderChatFeed();
    // Ping Ardour MCP asynchronously so UI renders immediately
    refreshArdourStatus();
    setInterval(refreshArdourStatus, 12000);
  }

  init().catch((err) => toast("Ошибка запуска UI: " + err.message, 5000));
})();
