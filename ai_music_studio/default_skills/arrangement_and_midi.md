---
name: Аранжировка и написание MIDI-партий
description: Создание нешаблонной аранжировки под референсы и голос через sfizz (AVL Drums / Pick Bass), DI-гитары + NAM, Surge XT и гуманизированный MIDI
category: arrangement
enabled: true
builtin: true
---

# Скилл: Аранжировка и живая MIDI-композиция (без клише!)

1. **Опора на «нейро-паспорт» референса и вокал пользователя:**
   - Изучи блок `neural_blueprint` у референсов и `unified_vocal_harmony` вокала (или вызови `studio_deep_analyze_reference` / `studio_analyze_vocal_fx`).
   - Пиши все гармонические инструменты в **единой согласованной тональности**, совместимой с контуром твоего вокала (`F0`), вместо случайных скачков по полутонам.
   - Повторяй **реальную ритм-сетку** ударных (`drum_pattern`: `kick_steps_16`, `snare_steps_16`, `hat_steps_16`), темп, смену плотности (`verse` -> `chorus`) и тембры инструментов референса.
   - **ЗАПРЕТ НА ШАБЛОНЫ:** НЕ вставляй по умолчанию «грустное пианино (Lonely Piano)», `alien-monolith-pad` или случайные `808`, если их нет в референсах или запросе!

2. **Живые инструменты вместо «пластикового GM»:**
   - Выбирай инструмент под реальный жанр через **`studio_inspect_sound_library`** (категории `sfz`, `sf2`, `surge`, `nam`, `all`) и ставь его одним вызовом **`studio_setup_track_instrument`**:
     - **Живые рок / панк / инди барабаны:** `plugin_type="sfizz"`, `preset_label="AVL Black Pearl 5pc"` (или `"AVL Red Zeppelin 5pc"`). Либо `FluidR3 GM` канал `9` (`midi_channel=9`, программы `16–18` Power Kits). Для электроники — `HS TR-808` / `TR-909`.
     - **Живой бас с медиатором (панк / пост-панк / рок):** `plugin_type="sfizz"`, `preset_label="Pick Bass"` (или `"Mute Pick Bass"`). Плюс по вкусу в инсерт `BOD_x64` (`TSE BOD`) или `.nam`!
     - **Живые электрогитары (пост-панк / гранж / инди):** `plugin_type="fluidsynth"`, `preset_label="FS Fender Telecaster DI"`, `program=0` (Standard) / `1` (Muted) / `2` (Muted 5ths & Solo), и **сразу после него на ту же дорожку добавь `NeuralAmpModeler.vst3`** (с `.nam` усилителем!) или `Airwindows Consolidated` + `TAL-Chorus-LX` / `Dragonfly`. Чистый `DI -> NAM` звучит как настоящая гитара через комбик!
     - **Синтезаторы / ключи / пады:** используй `Surge XT.vst3` (в библиотеке проиндексированы 3,500+ фабричных патчей), `TAL-NoiseMaker.vst3` или `Dexed.vst3` — выбирая конкретный пресет через Lua (`plug:get_info():get_presets(false)` -> `plug:load_preset`), а не дефолтную «пилу».

3. **Живой MIDI-грув (через `studio_compose_humanized_midi`):**
   - Используй **`studio_compose_humanized_midi`** вместо сухих квантованных нот по линейке:
     - задавай `role` (`drums`, `bass`, `guitar_strum`, `guitar_arp`, `lead`, `keys`);
     - для гитарного чёса/аккордов задавай `strum_ms=14..26` (имитация удара медиатором по струнам);
     - включай `humanize_timing_ms=10..18` и `humanize_velocity=10..16`, прописывай переходы/брейки перед припевом, октавные проходные ноты в басу и динамику куплета vs припева.
   - Включай дополнительные сэмплы из вкладки «Звуки» на нужных тактах через `ardour__audio_import_file`.
