#!/usr/bin/env bash
# OPEN_IN_TD.command
# Double-click this file in Finder to open the saved .toe in TouchDesigner.
#
# First-time setup:
#   1. Open TouchDesigner (blank project)
#   2. Press Alt+T to open the Textport
#   3. Paste and run this single line:
#
#      exec(open('/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/touchdesigner/td_auto_setup.py').read())
#
#   4. Wait ~5 seconds while the network builds and saves the .toe
#   5. After that, this .command file opens it directly.

TOE="/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/touchdesigner/topology_of_thought.toe"
TD_APP="/Applications/TouchDesigner.app"

if [ ! -f "$TOE" ]; then
    echo ""
    echo "  topology_of_thought.toe not found."
    echo ""
    echo "  First-time setup:"
    echo "  1. Open TouchDesigner"
    echo "  2. Press Alt+T (Textport)"
    echo "  3. Paste:"
    echo ""
    echo "     exec(open('$(dirname "$TOE")/td_auto_setup.py').read())"
    echo ""
    echo "  This builds the network and saves the .toe automatically."
    echo ""
    read -p "Press Enter to open TouchDesigner now..."
    open -a "$TD_APP"
    exit 0
fi

if [ ! -d "$TD_APP" ]; then
    echo "TouchDesigner not found at $TD_APP"
    echo "Install from https://derivative.ca/download"
    exit 1
fi

echo "Opening $TOE in TouchDesigner..."
open -a "$TD_APP" "$TOE"
