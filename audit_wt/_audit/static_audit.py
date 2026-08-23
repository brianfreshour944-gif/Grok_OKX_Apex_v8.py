"""
Static audit:
1. Await-target audit: for every `await <expr>` find what <expr> refers to
   (project function / external module attr / method) and whether it is a
   coroutine function (async def). Awaiting a non-awaitable is a TypeError
   at runtime, usually swallowed by broad except blocks.
2. Name-resolution audit: for every Name loaded in each module, check it is
   defined by an import, assignment, def/class, comprehension var, arg,
   global/nonlocal, or is a builtin. Flags undefined names (missing imports).
"""
import ast
import builtins
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILTINS = set(dir(builtins))

# ---------- pass 1: index all top-level & class-level defs per module ----------
module_defs = {}   # module_name -> {name: "async"|"sync"|"other"}
module_files = {}

for dirpath, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "_audit")]
    for f in sorted(files):
        if not f.endswith(".py"):
            continue
        path = os.path.join(dirpath, f)
        rel = os.path.relpath(path, ROOT)
        mod = os.path.splitext(rel.replace("\\", "/").replace("/", "."))[0]
        try:
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=rel)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in {rel}: {e}")
            continue
        module_files[mod] = (path, tree)
        defs = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "async" if isinstance(node, ast.AsyncFunctionDef) else "sync"
                defs.setdefault(node.name, kind)
            elif isinstance(node, ast.ClassDef):
                defs.setdefault(node.name, "class")
        module_defs[mod] = defs


def resolve_module(spec):
    """Map an import spec like 'config' or 'scripts.test_ws' to a known module."""
    if spec in module_defs:
        return spec
    # try suffixes (e.g., running from repo root, 'tests.conftest')
    parts = spec.split(".")
    for i in range(len(parts)):
        cand = ".".join(parts[i:])
        if cand in module_defs:
            return cand
    return None


print("=" * 78)
print("PASS 1: AWAIT-TARGET AUDIT")
print("=" * 78)
await_issues = 0
await_total = 0

for mod, (path, tree) in sorted(module_files.items()):
    # Track imports in this module: localname -> (module_spec, origname)
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports[a.asname or a.name.split(".")[0]] = (a.name, None)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imports[a.asname or a.name] = (node.module or "", a.name)

    # Functions defined in this module
    local_funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local_funcs[node.name] = "async" if isinstance(node, ast.AsyncFunctionDef) else "sync"

    for node in ast.walk(tree):
        if not isinstance(node, ast.Await):
            continue
        await_total += 1
        target = node.value
        desc = None
        status = None

        if isinstance(target, ast.Call):
            fn = target.func
        else:
            fn = target

        if isinstance(fn, ast.Name):
            name = fn.id
            if name in local_funcs:
                status = local_funcs[name]
                desc = f"{mod}.{name}() [local]"
            elif name in imports:
                modspec, orig = imports[name]
                tgt_mod = resolve_module(modspec)
                if tgt_mod:
                    kind = module_defs[tgt_mod].get(orig or name)
                    status = kind if kind else "unknown-in-module"
                    desc = f"{tgt_mod}.{orig or name} [imported]"
                else:
                    status = "external"
                    desc = f"{modspec}.{orig or name} [external lib]"
            else:
                status = "UNRESOLVED"
                desc = f"{name} [no import/def found]"
        elif isinstance(fn, ast.Attribute):
            val = fn.value
            base = None
            if isinstance(val, ast.Name):
                base = val.id
            if base and base in imports:
                modspec, _orig = imports[base]
                tgt_mod = resolve_module(modspec)
                attr = fn.attr
                if tgt_mod:
                    kind = module_defs[tgt_mod].get(attr)
                    status = kind if kind else "not-a-project-def"
                    desc = f"{tgt_mod}.{attr} [via {base}]"
                else:
                    status = "external"
                    desc = f"{modspec}.{attr} [external lib]"
            else:
                status = "method-or-unknown"
                desc = f"<expr>.{fn.attr}"
        else:
            status = "complex-expr"
            desc = ast.dump(fn)[:60]

        flag = ""
        if status == "sync":
            flag = "  <<<< AWAIT ON SYNC DEF"
            await_issues += 1
        elif status in ("UNRESOLVED", "not-a-project-def", "class"):
            flag = "  <<<< CHECK MANUALLY"
            await_issues += 1
        if flag or True:
            print(f"{os.path.relpath(path, ROOT)}:{node.lineno}: await -> {desc} [{status}]{flag}")

print(f"\nAwait sites scanned: {await_total}; flagged: {await_issues}")

print()
print("=" * 78)
print("PASS 2: NAME-RESOLUTION AUDIT (undefined-name detection)")
print("=" * 78)

class ScopeChecker(ast.NodeVisitor):
    """Collects bound names per scope; reports Load references never bound."""
    def __init__(self, path):
        self.path = path
        self.scopes = [set()]
        self.problems = []

    def bind(self, name):
        self.scopes[-1].add(name)

    def visit_FunctionDef(self, node):
        self.bind(node.name)
        self.scopes.append(set())
        for a in node.args.args + node.args.kwonlyargs + ([node.args.vararg] if node.args.vararg else []) + ([node.args.kwarg] if node.args.kwarg else []):
            if a: self.bind(a.arg)
        if node.args.posonlyargs:
            for a in node.args.posonlyargs: self.bind(a.arg)
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self.scopes.append(set())
        for a in node.args.args:
            self.bind(a.arg)
        self.visit(node.body)
        self.scopes.pop()

    def visit_ClassDef(self, node):
        self.bind(node.name)
        self.scopes.append(set())
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()

    def visit_Import(self, node):
        for a in node.names:
            self.bind(a.asname or a.name.split(".")[0])

    def visit_ImportFrom(self, node):
        for a in node.names:
            if a.name != "*":
                self.bind(a.asname or a.name)

    def visit_Global(self, node):
        for n in node.names:
            self.bind(n)

    def visit_Nonlocal(self, node):
        for n in node.names:
            self.bind(n)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if not any(node.id in s for s in self.scopes) and node.id not in BUILTINS:
                self.problems.append((node.lineno, node.id))
        elif isinstance(node.ctx, (ast.Store,)):
            self.bind(node.id)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bind(node.name)
        for stmt in node.body:
            self.visit(stmt)

    def visit_comprehension(self, node):
        # bind generator vars before visiting element conditions
        self.visit(node.target)
        self.visit(node.iter)
        for cond in node.ifs:
            self.visit(cond)

    def _comp(self, node, elts):
        self.scopes.append(set())
        for comp in node.generators:
            self.visit_comprehension(comp)
        for e in elts:
            self.visit(e)
        self.scopes.pop()

    def visit_ListComp(self, node):
        self._comp(node, [node.elt])

    def visit_SetComp(self, node):
        self._comp(node, [node.elt])

    def visit_GeneratorExp(self, node):
        self._comp(node, [node.elt])

    def visit_DictComp(self, node):
        self._comp(node, [node.key, node.value])


total_problems = 0
for mod, (path, tree) in sorted(module_files.items()):
    checker = ScopeChecker(path)
    checker.visit(tree)
    seen = set()
    for lineno, name in checker.problems:
        key = (lineno, name)
        if key in seen:
            continue
        seen.add(key)
        total_problems += 1
        print(f"{os.path.relpath(path, ROOT)}:{lineno}: possibly undefined name '{name}'")

print(f"\nUndefined-name candidates: {total_problems}")