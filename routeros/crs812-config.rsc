# RouterOS v7 config — MikroTik CRS812 DDQ (CRS812-8DS-2DQ-2DDQ-RM)
# Role: 2-node DGX Spark fabric, 400G->2x200G QSFP-DD breakout in cage 1.
# Apply in Winbox/SSH terminal. Persists automatically.

# --- 1. Lane grouping (CRITICAL) ---
# Each QSFP-DD cage exposes 8 lane-interfaces. An enabled interface claims all
# contiguous serdes lanes below the NEXT enabled interface. For 2x200G breakout
# you MUST enable ALL 8 lanes: dd-1-1 absorbs lanes 1-4, dd-1-5 absorbs 5-8.
# Enabling only dd-1-1 + dd-1-5 gives 50G per leg (the classic trap).

/interface ethernet disable [find name~"qsfp56-dd-1"]
/interface ethernet enable qsfp56-dd-1-1,qsfp56-dd-1-2,qsfp56-dd-1-3,qsfp56-dd-1-4,qsfp56-dd-1-5,qsfp56-dd-1-6,qsfp56-dd-1-7,qsfp56-dd-1-8

# FEC: leave auto (negotiates fec91 with CX7 DAC). Do NOT force advertise lists —
# `advertise=200G-baseCR4` alone produces EMPTY advertising + total no-link.

# --- 2. Bridge the two group masters (REQUIRED for L2) ---
# Default bridge only has ether1/ether2. Without this, links show 200G but ARP fails.
# Add ONLY the masters, never sub-lanes.

/interface bridge port add bridge=bridge interface=qsfp56-dd-1-1
/interface bridge port add bridge=bridge interface=qsfp56-dd-1-5

# Verify HW=yes (hardware offload):
# /interface bridge port print

# --- 3. Verify link state ---
# :put [/interface ethernet monitor qsfp56-dd-1-1 once as-value]
# :put [/interface ethernet monitor qsfp56-dd-1-5 once as-value]
# Want: status=link-ok  rate=200Gbps  fec=fec91  sfp-module-present=true

# --- 4. Security ---
# /user set admin password=YOUR_STRONG_PASSWORD
