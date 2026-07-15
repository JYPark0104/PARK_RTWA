import importlib
mods = ["open3d","sionna","tensorflow","trimesh","fastapi","uvicorn","numpy","mitsuba"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print("  %-12s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("  %-12s MISSING (%s)" % (m, type(e).__name__))
