---
name: Изучение и грамотное применение плагинов, инструментов и звуков
description: Обязательный протокол исследования плагинов, SFZ/SF2 инструментов, патчей Surge XT (3500+), NAM-моделей, сэмплов и вокальных FX перед настройкой
category: workflow
enabled: true
builtin: true
---

# Изучение и грамотное применение плагинов, инструментов и звуков

**Главное правило:** никогда не выбирай инструменты и не крути плагины вслепую! Сначала проверь библиотеку (`studio_inspect_sound_library`), карту параметров (`ardour__plugin_get_description`) и нейронный разбор референса (`studio_deep_analyze_reference`).

---

## 1. Твой доступный арсенал плагинов (VST3 / LV2 / VST2)

- **Живые инструменты и синты:**
  - **`sfizz` (`http://sfztools.github.io/sfizz`)** — сэмплер живых мультисэмпловых библиотек SFZ с готовыми пресетами: `"AVL Black Pearl 5pc"` / `"AVL Red Zeppelin 5pc"` (живые рок/инди барабаны), `"Pick Bass"` / `"Mute Pick Bass"` / `"Slap Bass"` (живой бас).
  - **`ACE Fluid Synth` (`lv2:urn:ardour:a-fluidsynth`)** — с готовыми пресетами: `"FS Fender Telecaster DI"` (реальные сэмплы электрогитары Fender Telecaster: prog 0 `Standard`, 1 `Muted`, 2 `Muted 5ths & Solo`, 3 `Solo & Muted 5ths`), `"FluidR3 GM"` (Prog `16–18` на 10 канале = `Power Drum Kits`, `8/30` Feedback Guitar), `"GeneralUser GS"`, `"HS TR-808"`, `"TR-909"`. *(Без загруженного пресета молчит!)*
  - **`Surge XT.vst3`** — мощнейший гибридный синтезатор (3,563 фабричных `.fxp` патча по категориям: Basses, Leads, Keys, Pads, Plucks, Chords, FX — ищи через `studio_inspect_sound_library(category="surge", query="...")`).
  - **`TAL-NoiseMaker.vst3` / `Dexed.vst3`** — аналоговый VA-синт и 6-операторный FM-синт DX7.

- **Гитарно-басовое усиление и сатурация:**
  - **`NeuralAmpModeler.vst3` (NAM)** + 7 `.nam` моделей в `samples/nam_models` (`Ceriatone_KingKong` Clean/Crunch/Lead, `Peavey_5150`, `Peavey_6505Plus`) — ставится **после `FS Fender Telecaster DI`** для настоящего лампового рок/пост-панк звука!
  - **`BOD_x64.dll` (TSE BOD v3)** — эмулятор SansAmp Bass Driver для плотного рычащего рок-баса.
  - **`Airwindows Consolidated.vst3`** — плёнка, сатураторы, консольные эмуляции (`ToTape`, `PurestDrive`, `Mackity`, `Console`, `Channel`, `Pressure`).

- **Вокальный питч, EQ, динамика и пространство:**
  - **`SnapTune.vst3`** — автотюн / питч-коррекция с привязкой к тональности песни.
  - **`TDR Nova.vst3`** — динамический EQ (де-эссер 5–8 кГц, подчистка мутности 200–400 Гц).
  - **`TDR VOS SlickEQ.vst3`**, **`TDR Molotok.vst3`**, **`TDR Kotelnikov.vst3`**, **`LoudMax.vst3`** — эквализация, характерная/мастеринговая компрессия и brickwall True-Peak лимитер (`-1.0 dBTP`).
  - **`DragonflyHallReverb` / `Plate` / `Room` / `EarlyReflections`**, **`TAL-Reverb-4.vst3`**, **`TAL-Chorus-LX.vst3`**, **`ykchorus.lv2`**, **`a-delay.lv2`**, **`Surge XT Effects.vst3`**.

---

## 2. Протокол настройки

1. После добавления/поиска плагина через `ardour__plugin_list_available` всегда вызывай `ardour__plugin_get_description`, проверяй реальные диапазоны `min..max` параметров и настраивай все ручки разом через **`studio_set_plugin_parameters`**.
2. Для инструментов используй **`studio_setup_track_instrument`**: она сама ставит `sfizz` / `ACE Fluid Synth` / VST3, грузит LV2-пресет, выставляет верный MIDI-канал и Bank/Program Change, а при желании сразу вешает `NAM` / `BOD` следом!
3. Пресеты плагина в Lua проверяй через `plug:get_info():get_presets(false)` и грузи через `plug:load_preset(ps)`.
4. Никогда не сдавай экспорт без контрольной проверки через **`studio_compare_mix_to_reference`** (`deep=true`).
