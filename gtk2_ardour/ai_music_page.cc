/*
 * Copyright (C) 2026 FL Radio / AI Music Studio Contributors
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifdef WAF_BUILD
#include "gtk2ardour-config.h"
#endif

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <glibmm/miscutils.h>
#include <glibmm/fileutils.h>
#include <glibmm/spawn.h>

#ifdef PLATFORM_WINDOWS
#include <windows.h>
#endif

#include "pbd/compose.h"
#include "pbd/error.h"
#include "pbd/file_utils.h"

#include "ardour/control_protocol_manager.h"
#include "ardour/session.h"

#include "gtkmm2ext/gtk_ui.h"
#include "gtkmm2ext/window_title.h"

#include "widgets/tooltips.h"

#include "ai_music_page.h"
#include "ardour_ui.h"
#include "keyboard.h"
#include "ui_config.h"

#include "pbd/i18n.h"

using namespace ARDOUR;
using namespace ArdourWidgets;
using namespace Gtkmm2ext;
using namespace Gtk;

#define PX_SCALE(px) std::max ((float)px, rintf ((float)px * UIConfiguration::instance ().get_ui_scale ()))

AiMusicPage::AiMusicPage ()
	: Tabbable (_("AI Music"), X_("ai_music"), NULL, true, Tabbable::NoPanes)
	, _center_align (0.5, 0.5, 0.0, 0.0)
	, _ai_music_button (_("AI Music"), ArdourButton::led_default_elements)
{
	/* Match Ardour's exact native widget style (Cairo ArdourButton) */
	_ai_music_button.set_name (X_("page switch button"));
	_ai_music_button.set_size_request ((int)PX_SCALE (220), (int)PX_SCALE (54));
	_ai_music_button.signal_clicked.connect (sigc::mem_fun (*this, &AiMusicPage::on_ai_music_button_clicked));
	ArdourWidgets::set_tooltip (
	    _ai_music_button,
	    _("Open AI Music Studio window (full AI DAW control, references, vocals, lyrics, skills & OpenAI-compatible providers)"));

	_center_box.set_spacing (0);
	_center_box.pack_start (_ai_music_button, false, false);
	_center_align.add (_center_box);

	/* Pack native Ardour ApplicationBar + single-button content area */
	content_app_bar.add (_application_bar);
	content_main.add (_center_align);

	_ai_music_button.show ();
	_center_box.show ();
	_center_align.show ();

	update_title ();
}

AiMusicPage::~AiMusicPage ()
{
}

void
AiMusicPage::set_session (ARDOUR::Session* s)
{
	SessionHandlePtr::set_session (s);
	_application_bar.set_session (s);
	update_title ();
}

void
AiMusicPage::session_going_away ()
{
	SessionHandlePtr::session_going_away ();
	update_title ();
}

void
AiMusicPage::focus_on_clock ()
{
}

void
AiMusicPage::update_title ()
{
	if (!own_window ()) {
		return;
	}
	WindowTitle title (Glib::get_application_name ());
	title += _("AI Music");
	if (_session) {
		title += _session->snap_name ();
	}
	own_window ()->set_title (title.get_string ());
}

Gtk::Window*
AiMusicPage::use_own_window (bool and_fill_it)
{
	bool         new_window = !own_window ();
	Gtk::Window* win        = Tabbable::use_own_window (and_fill_it);

	if (win && new_window) {
		win->set_name ("AiMusicWindow");
		ARDOUR_UI::instance ()->setup_toplevel_window (*win, _("AI Music"), this);
		win->signal_event ().connect (sigc::bind (sigc::ptr_fun (&Keyboard::catch_user_event_for_pre_dialog_focus), win));
		update_title ();
	}

	contents ().show ();
	return win;
}

XMLNode&
AiMusicPage::get_state () const
{
	XMLNode* node = new XMLNode (X_("AiMusicPage"));
	node->add_child_nocopy (Tabbable::get_state ());
	return *node;
}

int
AiMusicPage::set_state (const XMLNode& node, int version)
{
	return Tabbable::set_state (node, version);
}

void
AiMusicPage::ensure_mcp_http_active ()
{
	if (!_session) {
		return;
	}

	ControlProtocolManager& cpm = ControlProtocolManager::instance ();
	for (auto* cpi : cpm.control_protocol_infos ()) {
		if (!cpi) {
			continue;
		}
		if (cpi->name.find ("MCP HTTP") != std::string::npos) {
			if (!cpi->active ()) {
				cpm.activate (*cpi, cpi->config);
			}
			break;
		}
	}
}

void
AiMusicPage::launch_ai_music_studio ()
{
	/* Resolve parent workspace directory containing ai_music_studio */
	std::string studio_parent;

	const char* env_dir = std::getenv ("AI_MUSIC_STUDIO_DIR");
	if (env_dir && *env_dir && Glib::file_test (env_dir, Glib::FILE_TEST_IS_DIR)) {
		studio_parent = Glib::path_get_dirname (env_dir);
	}

	if (studio_parent.empty ()) {
		std::vector<std::string> candidates;
		std::string              cwd = Glib::get_current_dir ();
		candidates.push_back (cwd);
		candidates.push_back (Glib::path_get_dirname (cwd));
#ifdef PLATFORM_WINDOWS
		char exe_buf[MAX_PATH] = {0};
		if (GetModuleFileNameA (NULL, exe_buf, MAX_PATH) > 0) {
			std::string exe_dir = Glib::path_get_dirname (exe_buf);
			candidates.push_back (exe_dir);
			candidates.push_back (Glib::path_get_dirname (exe_dir));
			candidates.push_back (Glib::path_get_dirname (Glib::path_get_dirname (exe_dir)));
		}
#endif
		for (const auto& base : candidates) {
			std::string probe = Glib::build_filename (base, "ai_music_studio");
			if (Glib::file_test (Glib::build_filename (probe, "launcher.py"), Glib::FILE_TEST_EXISTS)) {
				studio_parent = base;
				break;
			}
		}
	}

	const std::string active_theme = UIConfiguration::instance ().get_color_file ();

#ifdef PLATFORM_WINDOWS
	if (!active_theme.empty ()) {
		SetEnvironmentVariableA ("AI_MUSIC_THEME", active_theme.c_str ());
	}
	if (!studio_parent.empty ()) {
		SetEnvironmentVariableA ("PYTHONPATH", studio_parent.c_str ());
	}

	std::string cmd = "pythonw -m ai_music_studio.launcher";
	STARTUPINFOA        si;
	PROCESS_INFORMATION pi;
	ZeroMemory (&si, sizeof (si));
	si.cb = sizeof (si);
	ZeroMemory (&pi, sizeof (pi));

	std::vector<char> cmd_mut (cmd.begin (), cmd.end ());
	cmd_mut.push_back ('\0');

	BOOL ok = CreateProcessA (
	    NULL,
	    cmd_mut.data (),
	    NULL,
	    NULL,
	    FALSE,
	    DETACHED_PROCESS | CREATE_NO_WINDOW,
	    NULL,
	    studio_parent.empty () ? NULL : studio_parent.c_str (),
	    &si,
	    &pi);

	if (!ok) {
		std::string fallback = "python -m ai_music_studio.launcher";
		std::vector<char> fb_mut (fallback.begin (), fallback.end ());
		fb_mut.push_back ('\0');
		ok = CreateProcessA (
		    NULL,
		    fb_mut.data (),
		    NULL,
		    NULL,
		    FALSE,
		    DETACHED_PROCESS | CREATE_NO_WINDOW,
		    NULL,
		    studio_parent.empty () ? NULL : studio_parent.c_str (),
		    &si,
		    &pi);
	}

	if (ok) {
		CloseHandle (pi.hProcess);
		CloseHandle (pi.hThread);
	} else {
		PBD::error << _("AI Music: failed to launch ai_music_studio.launcher") << endmsg;
	}
#else
	try {
		std::vector<std::string> argv = {"python3", "-m", "ai_music_studio.launcher"};
		std::vector<std::string> envp = Glib::listenv ();
		if (!studio_parent.empty ()) {
			envp.push_back ("PYTHONPATH=" + studio_parent);
		}
		if (!active_theme.empty ()) {
			envp.push_back ("AI_MUSIC_THEME=" + active_theme);
		}
		Glib::spawn_async (
		    studio_parent.empty () ? Glib::get_current_dir () : studio_parent,
		    argv,
		    envp,
		    Glib::SPAWN_SEARCH_PATH);
	} catch (const Glib::Error& err) {
		PBD::error << string_compose (_("AI Music: failed to launch studio (%1)"), err.what ()) << endmsg;
	}
#endif
}

void
AiMusicPage::on_ai_music_button_clicked ()
{
	_ai_music_button.set_active_state (Gtkmm2ext::ImplicitActive);
	ensure_mcp_http_active ();
	launch_ai_music_studio ();
}
