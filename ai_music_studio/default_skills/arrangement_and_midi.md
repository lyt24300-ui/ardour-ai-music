---
name: Аранжировка и написание MIDI-партий
description: Создание структуры трека, маркеров, ударных, баса, аккордов и мелодий через Ardour MIDI JSON
category: arrangement
enabled: true
builtin: true
---

# Скилл: Аранжировка и MIDI-композиция

1. **Настройка сетки и структуры:**
   - Выстави темп (`tempo/set_bpm`) и размер (`tempo/set_meter`) по референсу или запросу пользователя.
   - Создай маркеры секций (`markers/add`): `Intro`, `Verse 1`, `Pre-Chorus`, `Chorus 1`, `Verse 2`, `Chorus 2`, `Bridge`, `Outro` с учётом текста песни.
2. **Подбор инструментов и обязательная настройка SoundFont (`.sf2`):**
   - Проверь доступные плагины через `plugin_list_available`, а также `soundfonts` и `fluid_synth_presets` в контексте сессии.
   - **ВАЖНО:** `ACE Fluid Synth` (`lv2:urn:ardour:a-fluidsynth`) без загруженного `.sf2` файла **молчит**!
   - Если в `fluid_synth_presets` есть пресет (например `"GeneralUser GS"`), сразу после добавления `ACE Fluid Synth` на трек загрузи пресет и отправь нужную GM-программу через проверенный `lua_run_script` (`plug:preset_by_label("GeneralUser GS")` -> `plug:load_preset(ps)` + `ARDOUR.RawMidiParser()` + `ins:write_immediate_event(...)`).
   - Если `.sf2` банков и сторонних VSTi нет — сразу предложи карточку `studio_request_plugin_install`, либо по согласованию поставь встроенный `ACE Reasonable Synth` (`lv2:https://community.ardour.org/node/7596`).
   - Не трать ходы на угадывание несуществующих Lua-методов вроде `region_by_id`, `new_patch_change_command` или `PatchChangePtr`.
3. **Пакетная генерация музыкальных партий (`midi_note/import_json_*`):**
   - Пиши гармонически согласованные партии строго в выбранной тональности и ладу (Key/Scale).
   - Для ударных используй `is_drum_mode: true` (канал 10, стандартная GM-раскладка: Kick=36, Snare=38, Clap=39, Closed Hat=42, Open Hat=46), варьируй `velocity` для живого грува.
   - Для баса, аккордов и лидов задавай длительность `l` и динамику `velocity`; используй `{ "bar": N, "repeat": M }` для компактного повторения паттернов.
   - Обязательно включай в аранжировку дополнительные сэмплы из вкладки «Звуки» с помощью `audio/import_file` на нужных тактах.
