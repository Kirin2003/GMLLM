from pathlib import Path
import ast
import astpretty

# 定义路径
path = Path("/Data2/hxq/datasets/incremental_packages/2022-01/malicious/AadhaarCrypt-1.0/setup.py")

src = path.read_text(encoding="utf-8", errors="ignore")

print("=" * 60)
print("1. 源代码")
print("=" * 60)
print(src)

# 解析为AST
tree = ast.parse(src)

print("\n" + "=" * 60)
print("2. AST dump (完整结构)")
print("=" * 60)
print(astpretty.pprint(tree))

print("\n" + "=" * 60)
print("3. Module 节点的直接子节点")
print("=" * 60)
print(f"tree 类型: {type(tree)}")  # <class 'ast.Module'>
print(f"body 长度: {len(tree.body)}")
for i, node in enumerate(tree.body):
    print(f"  [{i}] {type(node).__name__} (行 {node.lineno})")

print("\n" + "=" * 60)
print("4. 遍历所有节点 (walk)")
print("=" * 60)
for i, node in enumerate(ast.walk(tree)):
    print(f"  [{i}] {type(node).__name__}")

print("\n" + "=" * 60)
print("5. 使用 NodeVisitor 自定义遍历")
print("=" * 60)

class MyVisitor(ast.NodeVisitor):
    def visit_Module(self, node):
        print(f"访问 Module: {type(node).__name__}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        print(f"  函数定义: {node.name} (行 {node.lineno})")
        print(f"    参数: {[arg.arg for arg in node.args.args]}")
        print(f"    函数体语句数: {len(node.body)}")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        print(f"  类定义: {node.name} (行 {node.lineno})")
        print(f"    基类: {[b.id if hasattr(b, 'id') else astpretty.pprint(b) for b in node.bases]}")
        print(f"    类体语句数: {len(node.body)}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            print(f"  函数调用: {node.func.id} (行 {node.lineno})")
        elif isinstance(node.func, ast.Attribute):
            print(f"  方法调用: {astpretty.pprint(node.func)} (行 {node.lineno})")
        self.generic_visit(node)

visitor = MyVisitor()
visitor.visit(tree)
