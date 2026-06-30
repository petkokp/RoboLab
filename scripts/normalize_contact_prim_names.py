"""Rename USD prims that share their contact-filter root's name.

In isaacsim6, when a contact-filter target (scene/<obj>) has a descendant also named <obj>,
PhysX returns a backend-less contact view and isaaclab3 crashes in ContactSensor._create_buffers.
Rename any descendant primSpec matching an ancestor's name to <name>_u<n> (deepest first),
per layer so each file edits only its own specs. Idempotent.

    python scripts/normalize_contact_prim_names.py [ROOT ...] [--apply]
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
    return spec.hasPayloads or spec.hasReferences


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
    stage = Usd.Stage.Open(f)
    layer = stage.GetRootLayer()             # this file's OWN specs only (no composed asset prims)
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
