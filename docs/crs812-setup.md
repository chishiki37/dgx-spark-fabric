# MikroTik CRS812 DDQ Fabric Setup for DGX Spark Clusters

Verified Jul 31, 2026 on Vikas's CRS812 DDQ (CRS812-8DS-2DQ-2DDQ-RM, RouterOS v7, Marvell switch chip) with 2× DGX Spark via NADDOD `Q2Q56-400G-CU1` (QSFP-DD 400G → 2× QSFP56 200G passive DAC breakout, 1m, CMIS 5.0). **End state: both lanes at `rate=200Gbps`, `link-ok`, fec91 on switch; `Speed: 200000Mb/s` on both Sparks.** ✅

## Management Access (switch has NO LAN connection)

The CRS812 in a fabric role has no presence on the home LAN. To reach RouterOS:

1. **Patch cable: switch RJ45 MGMT port ↔ Mac USB-Ethernet dongle (or Spark `enP7s7`).** Set host to static `192.168.88.10/24`, then `ssh admin@192.168.88.1` (or WebFig `https://192.168.88.1` → New Terminal).
2. **⚠️ The RJ45 port labeled CONSOLE is SERIAL (RS-232), not Ethernet.** Plugging Ethernet in shows no link, forever — it looks identical to MGMT. This mixup cost real time. MGMT is the port whose LED lights (amber on CRS812 MGMT port 1).
3. Factory IP is `192.168.88.1` **even when its DHCP server is OFF** — Mac will self-assign 169.254.x.x; don't be fooled, set static and the switch still answers at .1.
4. **Default admin password is printed on the chassis sticker** (RouterOS v7 per-device). User `admin`. Change it after first login (`/user set admin password=...`).
5. SSH from new macOS OpenSSH may need `-o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedKeyTypes=+ssh-rsa`.
6. Serial console fallback (always works): USB-serial adapter + Cisco-style RJ45 console cable, `screen /dev/cu.usbserial-* 115200`.

## QSFP-DD Lane Grouping — THE Key Lesson (empirically determined)

RouterOS exposes each QSFP-DD cage as **8 lane-interfaces**: `qsfp56-dd-<cage>-1` … `-8`.

**An enabled interface claims all contiguous serdes lanes below the next enabled interface.** Measured on cage 1:

| Enabled set | Result |
|---|---|
| `dd-1-1` alone | 1 lane → **50G-CR** |
| `dd-1-1` + `dd-1-2` | dd-1-1 gets 2 lanes → **100G-CR2** |
| `dd-1-1` + `dd-1-5` only | both stuck at 1 lane → **50G each** ❌ **THE TRAP** — looks like the natural "2×200G" combo (lanes 1/5 are the only ones advertising 200G/400G modes) but is NOT |
| **ALL 8 (`dd-1-1`…`dd-1-8`)** | dd-1-1 gets lanes 1-4, dd-1-5 gets lanes 5-8 → **200G-CR4 on both legs** ✅ |

**For a 2×200G breakout: enable ALL 8 lane interfaces of the cage.** Traffic flows only on the two group masters (`dd-1-1`, `dd-1-5`); lanes 2-4/6-8 are grouping fodder, not endpoints. (Default factory state = everything enabled across all cages, which is why links came up at all after reseating — the original no-link was a seating issue, not config.)

Config:

```
/interface ethernet disable [find name~"qsfp56-dd-1"]
/interface ethernet enable qsfp56-dd-1-1,qsfp56-dd-1-2,qsfp56-dd-1-3,qsfp56-dd-1-4,qsfp56-dd-1-5,qsfp56-dd-1-6,qsfp56-dd-1-7,qsfp56-dd-1-8
```

RouterOS persists config across reboots automatically. Verify (per-lane; the 2-column monitor truncates):

```
:put [/interface ethernet monitor qsfp56-dd-1-1 once as-value]
:put [/interface ethernet monitor qsfp56-dd-1-5 once as-value]
```

Want: `status=link-ok`, `rate=200Gbps`, `fec=fec91`. The plain `qsfp56-1-x`/`qsfp56-2-x` 200G cages follow the same pattern with 4 lanes each.

## RouterOS v7 Quirks (this build)

- **`fec-mode` enum is `auto|fec74|fec91|off`** — `rs-fec` is a syntax error (error column points at the value). fec91 = RS-FEC/Clause 91. `auto` works fine for 200G-CR4 DAC (negotiated FEC ends up fec91 anyway); forcing fec91 neither helped nor blocked.
- **No `speed=` property** on these lane interfaces. Restrict modes via `advertise=` — but ⚠️ **forcing `advertise=200G-baseCR4` alone produces EMPTY advertising + total no-link** (invalid AN combo on Marvell). Restore with the full CR list:
  ```
  /interface ethernet set qsfp56-dd-1-1,qsfp56-dd-1-5 advertise=10G-baseCR,40G-baseCR4,25G-baseCR,50G-baseCR2,100G-baseCR4,50G-baseCR,100G-baseCR2,200G-baseCR4
  ```
- `set` with a comma-list + property can throw a syntax error — set ports one at a time if so.
- After changing the enabled-lane set: `:delay 8` before monitoring (training takes seconds).
- Tab-complete everything (`set qsfp56-dd-1-1 fec-mode=` + Tab) — enum names vary across builds.

## Physical Debug Sequence (all hit in one session)

1. Spark `ethtool`: "No cable" on all ports → fan-out end not seated in the Spark. Push until click.
2. "Autoneg, No partner detected" → cable seated, switch side not negotiating (lane config, or…)
3. `sfp-module-present: no` in monitor → **DD end not seated in the switch cage**. QSFP-DD latches can feel closed while 1-2mm short: pull the bail, reinsert until a distinct CLICK, push once more. EEPROM (vendor/part/CMIS rev) appears within seconds of a good seat.
4. Module present but `rate=50Gbps` → lane-grouping issue (section above), NOT FEC and NOT the NIC: CX7 advertises `200000baseCR4/Full` in both Supported AND Advertised (verify: `ethtool <iface> | sed -n '/Advertised link modes/,/Advertised pause/p'`).

## Topology Verification — Don't Assume, Read the MAC Table

A 2×200G breakout has exactly **2 legs**. If `ethtool` shows more cabled ports across the Sparks than legs, your mental topology is wrong — trace physically. Definitive lane↔NIC map:

1. Assign IPs on the Spark fabric ports (`sudo nmcli device set <iface> managed no`, `sudo ip addr add …/24 dev <iface>`, link up)
2. Ping across, then on the switch: `/interface bridge host print` — learned MACs on `dd-1-1`/`dd-1-5` identify which NIC is behind each lane. Spark NIC MACs: `ip link show <iface> | grep ether`.

## Dual-Rail & IPs

Each Spark has 2 CX7 cards: Card 1 = domain 0000 (`enp1s0f*`, lowercase p), Card 2 = domain 0002 (`enP2p1s0f*`, capital P). One breakout = one 200G rail per Spark. A second breakout in cage 2 (`qsfp56-dd-2-*`, same all-8-lanes rule) adds rail 2. Suggested scheme: rail 1 (Card 1) = 10.10.10.x/24 (matches existing recipes), rail 2 (Card 2) = 10.10.20.x/24. NCCL filter still applies: `NCCL_IB_HCA=rocep1s0f` (lowercase p) selects Card 1 RDMA devices only.

CX7 NICs still need the PCIe hotplug rescan after boot (`debug_state` + rescan — see multi-node-setup-guide.md), unchanged by the switch.

## RoCE Through the Switch: CONFIRMED (Jul 31, 2026)

- `ib_write_bw -d rocep1s0f0 -s 1048576 -q 4` → **13.3 GB/s**, no `ibv_modify_qp` error 61 ✅
- NCCL bench (`scripts/nccl-bench.py` in `--privileged` eugr-nightly container): **BusBW 40-51 GB/s via `NET/IB/0`** with GPUDirect RDMA ✅
- TCP iperf3 8-stream: **~107 Gbit/s per rail** (ARM CPU-bound; direct link was ~15 Gb/s)
- PFC/ECN NOT needed for this result — plain L2, MTU 1500, fec91. (PFC/ECN may add more; untested.)
- Container requirement: `--privileged` (or `/dev/infiniband` mapped) or NCCL says `NET/IB : No device found`. NCCL 2.30: `NCCL_NET=IB` is fatal-if-unavailable — omit it and pin `NCCL_IB_HCA` instead.

## End-to-End Decode: CRS812 RoCE vs Direct Link (Jul 31, 2026)

Baselines = Cluster Manager fleet runs (Jul 22-23, direct link); new = same models relaunched over the fabric (400-token completions, temp 0):

| Model | Direct link (Jul 22-23) | CRS812 RoCE (Jul 31) | Delta |
|---|---|---|---|
| MiniMax M2.7 AWQ (eugr-nightly) | ~32 tok/s (RoCE direct) | **40.2 tok/s** | **+26%** |
| DeepSeek V4 Flash FP8 (dsv4 fork, MTP spec-2) | ~30 tok/s (Socket) | **39.2 tok/s** | **+31%** |
| MiMo V2.5 NVFP4 (mimo container + mods) | ~19 tok/s (Socket) | **25.4 tok/s** | **+33%** |

Pattern: uniform +25-33% across all three — the direct link (Socket, or RoCE with the error-61 workarounds) taxed every model by ~a quarter of decode speed; the CRS812 removes that tax regardless of model architecture. (Older Jun/Jul-22 references: MiniMax 32.7 RoCE direct, DSV4 37.5 with earlier MTP config — superseded by the Jul 22-23 fleet numbers.)

Launches: MiniMax via manual-2node script (`TOPOLOGY=crs812`); DSV4/MiMo via spark-cluster-manager Ray scripts patched for RoCE (`--privileged`, `NCCL_IB_HCA=rocep1s0f0`, `NCCL_IB_GID_INDEX=3`, iface `enp1s0f0np0` on both nodes).

## Open Items

- Persistence on Sparks: extend the `cx7-direct-link.service` systemd units with the switch-port interface names/IPs (both cards if dual-rail).
- PFC/ECN on the switch ports for possibly-more RoCE throughput — untested.
- Second-rail mystery: Card 2 ports showed 200G link while cage 2 was empty — trace physically (likely a second breakout in the plain `qsfp56-1/2` 200G cages).
