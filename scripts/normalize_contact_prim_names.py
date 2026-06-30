"""Universal, permanent fix for the isaacsim6/isaaclab3 contact-view backend crash.

Invariant enforced: no descendant prim may share the name of its TOP-LEVEL (subtree-root)
prim -- i.e. the prim that a contact filter targets (asset defaultPrim, or scene/<obj>).
(Deeper duplicates that don't match the object-root name -- e.g. snickers_bar's collider
meshes, apple_01's Looks/apple -- are harmless and left untouched; verified apple_01 binds a
contact backend despite its deep duplicate.)

Why: PhysX builds each contact sensor's filtered contact view by matching the filter prim
path against the stage. When a contact filter target (a dynamic object at scene/<obj>) has a
DESCENDANT prim also named <obj>, PhysX returns a backend-less contact view and isaaclab3
crashes in ContactSensor._create_buffers (AttributeError on filter_count). That duplication
comes from BOTH layers:
  - object USDs whose defaultPrim self-nests:  /plate_small/plate_small/plate_small
  - scene USDs with stale/orphaned overrides that recreate the name as typeless prims:
      def "plate_small" (payload=...) { over "plate_small_inst" { over "plate_small" {
        over "plate_small" {...} } } }    (the asset no longer has those inner prims, so the
        override only materializes empty typeless prims -- it moves nothing real)

This script operates per-LAYER (not on the composed stage), so each file only edits its OWN
specs: asset files fix asset self-nesting; scene files fix their own orphaned overrides. It
renames any primSpec whose name equals an ancestor primSpec's name to `<name>_u<n>` (deepest
first; Sdf.BatchNamespaceEdit relocates internal targets). Top-level object prims (whose name
the contact filter targets, e.g. scene/plate_small) are never renamed -- only descendants that
duplicate an ancestor name. Idempotent.

Usage:
    python scripts/normalize_contact_prim_names.py [ROOT ...] [--apply]
    (default ROOTs: assets/objects and assets/scenes relative to repo root)
"""
import glob
import os
import sys

from pxr import Sdf, Usd

_here = os.path.dirname(os.path.abspath(__file__))
_default_roots = [os.path.join(_here, "..", "assets", "objects"),
                  os.path.join(_here, "..", "assets", "scenes")]
APPLY = "--apply" in sys.argv
roots = [a for a in sys.argv[1:] if not a.startswith("-")] or _default_roots

files = []
for r in roots:
    files += glob.glob(os.path.join(r, "**", "*.usd"), recursive=True)
    files += glob.glob(os.path.join(r, "**", "*.usda"), recursive=True)
files = sorted(set(files))


def descendants_named(spec, name, out):
    for child in spec.nameChildren:
        if child.name == name:
            out.append(child.path)
        descendants_named(child, name, out)


def has_asset_arc(spec):
    # an "object root" instantiates an asset (payload or reference)
    try:
        if spec.hasPayloads or spec.hasReferences:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        if spec.payloadList.GetAddedOrExplicitItems() or spec.referenceList.GetAddedOrExplicitItems():
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def object_root_specs(stage, layer):
    """Object roots = the prims a contact filter can target: the defaultPrim (object asset
    files) plus every payload/reference-bearing prim (scene files instantiate objects that way)."""
    roots, seen = [], set()
    dp = stage.GetDefaultPrim()
    if dp:
        ds = layer.GetPrimAtPath(dp.GetPath())
        if ds:
            roots.append(ds); seen.add(ds.path)

    def walk(spec):
        if has_asset_arc(spec) and spec.path not in seen:
            roots.append(spec); seen.add(spec.path)
        for c in spec.nameChildren:
            walk(c)

    for r in layer.rootPrims:
        walk(r)
    return roots


total = 0
for f in files:
    try:
        stage = Usd.Stage.Open(f)            # robust format detection (Sdf.FindOrOpen is not)
        layer = stage.GetRootLayer()         # this file's OWN specs only (no composed asset prims)
    except Exception as e:  # noqa: BLE001
        print(f"OPEN_ERR\t{f}\t{e!r}"); continue
    if layer is None:
        continue
    dups, seen = [], set()
    for rspec in object_root_specs(stage, layer):
        tmp = []
        descendants_named(rspec, rspec.name, tmp)
        for p in tmp:
            if p not in seen:
                seen.add(p); dups.append(p)
    if not dups:
        continue
    dups.sort(key=lambda p: p.pathString.count("/"), reverse=True)  # deepest first
    rel = f
    if not APPLY:
        for p in dups:
            print(f"WOULD_FIX\t{rel}\t{p}")
        total += len(dups)
        continue
    edit = Sdf.BatchNamespaceEdit()
    cnt = {}
    for p in dups:
        nm = p.name
        cnt[nm] = cnt.get(nm, 0) + 1
        edit.Add(p.pathString, p.GetParentPath().AppendChild(f"{nm}_u{cnt[nm]}").pathString)
    if layer.Apply(edit):
        layer.Save()
        print(f"FIXED\t{rel}\trenamed {len(dups)} prim(s)")
        total += len(dups)
    else:
        print(f"FAIL\t{rel}\t{len(dups)} prim(s) (namespace edit rejected)")

print(f"DONE\trenamed_prims={total}\tapply={APPLY}\tfiles_scanned={len(files)}")
