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

#pragma once

#include <ytkmm/alignment.h>
#include <ytkmm/box.h>

#include "ardour/session_handle.h"

#include "widgets/ardour_button.h"
#include "widgets/tabbable.h"

#include "application_bar.h"

class AiMusicPage : public ArdourWidgets::Tabbable, public ARDOUR::SessionHandlePtr, public PBD::ScopedConnectionList
{
public:
	AiMusicPage ();
	~AiMusicPage ();

	void set_session (ARDOUR::Session*);

	XMLNode& get_state () const;
	int      set_state (const XMLNode&, int /* version */);

	Gtk::Window* use_own_window (bool and_fill_it);

	void focus_on_clock ();

private:
	void update_title ();
	void session_going_away ();
	void on_ai_music_button_clicked ();
	void ensure_mcp_http_active ();
	void launch_ai_music_studio ();

	ApplicationBar            _application_bar;
	Gtk::Alignment            _center_align;
	Gtk::VBox                 _center_box;
	ArdourWidgets::ArdourButton _ai_music_button;
};
