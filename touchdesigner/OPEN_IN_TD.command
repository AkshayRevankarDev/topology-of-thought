#!/usr/bin/env bash
# OPEN_IN_TD.command
# Double-click in Finder to open the Topology of Thought TD project.
#
# First-time setup:
#   1. From a terminal at the project root:
#        ./setup_td.sh                # creates .venv_td with Python 3.11
#   2. Open TouchDesigner once, then go to:
#        Edit → Preferences → DATs → Python 64-bit Module Path
#      and paste the path printed by setup_td.sh.  Restart TD.
#   3. Open a blank project in TD, press Alt+T (Textport), and paste:
#
#        exec(open(project.folder + '/td_setup.py').read())
#
#      (or use the absolute path to td_setup.py from this repo.)
#      Wait ~5 seconds for the network to build and save the .toe.
#   4. After that, this .command file opens the saved .toe directly.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOE="$SCRIPT_DIR/topology_of_thought.toe"
TD_APP="/Applications/TouchDesigner.app"

if [ ! -d "$TD_APP" ]; then
    echo "TouchDesigner not found at $TD_APP"
    echo "Install from https://derivative.ca/download"
    exit 1
fi

if [ ! -f "$TOE" ]; then
    echo ""
    echo "  topology_of_thought.toe not found at:"
    echo "    $TOE"
    echo ""
    echo "  First-time setup:"
    echo "  1. From a terminal at the project root:"
    echo "       cd \"$PROJECT_ROOT\""
    echo "       ./setup_td.sh"
    echo "  2. Open TouchDesigner, press Alt+T (Textport), paste:"
    echo ""
    echo "       exec(open('$PROJECT_ROOT/td_setup.py').read())"
    echo ""
    echo "  This builds the network and saves the .toe automatically."
    echo ""
    read -p "Press Enter to open TouchDesigner now..."
    open -a "$TD_APP"
    exit 0
fi

echo "Opening $TOE in TouchDesigner..."
open -a "$TD_APP" "$TOE"
