#!/bin/bash
# Download the public maintenance manuals the chatbot answers from.
# Run from the gauge folder:  bash get_manuals.sh
# PDFs stay local (keep manuals/ out of the public git repo: they belong to their publishers).
set -e
cd "$(dirname "$0")"
mkdir -p manuals/pump manuals/fan manuals/valve manuals/all
UA="Mozilla/5.0 (X11; Linux aarch64)"
get() { echo "-> $2"; wget -q --user-agent="$UA" -O "$2" "$1" || curl -sL -A "$UA" -o "$2" "$1"; }

get "https://www.peerlesspump.com/wp-content/uploads/2020/08/2880549_Horizontal-Centrifugal-Pumps_Word.pdf" manuals/pump/peerless_horizontal_centrifugal_iom.pdf
get "https://www.energy.gov/sites/prod/files/2014/05/f16/pump.pdf"                                         manuals/pump/doe_pumping_system_sourcebook.pdf
get "https://content.greenheck.com/public/DAMProd/Original/10010/479870USF_iom.pdf"                          manuals/fan/greenheck_usf_fan_iom.pdf
get "https://docs.nlr.gov/docs/fy03osti/29166.pdf"                                                           manuals/fan/doe_fan_system_sourcebook.pdf
get "https://assets.unilogcorp.com/267/ITEM/DOC/Asco_EF8210G35_120_Instruction_Installation_Manual.pdf"   manuals/valve/asco_8210_solenoid_valve_im.pdf
get "https://www.osha.gov/sites/default/files/publications/OSHAFS3529.pdf"                                   manuals/all/osha_lockout_tagout_factsheet.pdf

cat > manuals/sources.json << 'EOF'
[
 {"file": "pump/peerless_horizontal_centrifugal_iom.pdf", "machine": "pump",  "title": "Peerless Pump: Horizontal Centrifugal Pumps IOM"},
 {"file": "pump/doe_pumping_system_sourcebook.pdf",       "machine": "pump",  "title": "US DOE: Improving Pumping System Performance"},
 {"file": "fan/greenheck_usf_fan_iom.pdf",                "machine": "fan",   "title": "Greenheck: USF Utility Fan IOM"},
 {"file": "fan/doe_fan_system_sourcebook.pdf",            "machine": "fan",   "title": "US DOE: Improving Fan System Performance"},
 {"file": "valve/asco_8210_solenoid_valve_im.pdf",       "machine": "valve", "title": "ASCO: 8210 Series Solenoid Valve I&M"},
 {"file": "all/osha_lockout_tagout_factsheet.pdf",        "machine": "all",   "title": "OSHA: Lockout/Tagout Fact Sheet"}
]
EOF
ls -la manuals/*/*.pdf
echo "Check every PDF above is bigger than ~50 KB (a tiny file means the download was blocked)."
