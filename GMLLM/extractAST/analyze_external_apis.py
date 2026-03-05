import ast
import os
from pathlib import Path

class ExternalAPIAnalyzer(ast.NodeVisitor):
    def __init__(self, package_name):
        self.package_name = package_name
        self.local_defs = set()       # 记录当前文件内定义的函数/类名
        self.imports = {}             # 记录导入映射 {本地名称: 真实来源模块}
        self.assignments = {}         # 【新增】记事本：记录变量赋值 {变量名: 来源API}
        self.external_calls = set()   # 保存最终的外部 API 调用

    # --- 1. 收集当前文件内的自定义定义 ---
    def visit_FunctionDef(self, node):
        self.local_defs.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.local_defs.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.local_defs.add(node.name)
        self.generic_visit(node)

    # --- 2. 收集并标记导入的来源 ---
    def visit_Import(self, node):
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.imports[local_name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        is_internal = node.level > 0 or module.startswith(self.package_name)
        
        for alias in node.names:
            local_name = alias.asname or alias.name
            if is_internal:
                self.imports[local_name] = "__INTERNAL__"
            else:
                self.imports[local_name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    # --- 3. 【新增】捕获赋值操作，进行轻量级变量追踪 ---
    def visit_Assign(self, node):
        # 尝试提取等号右边（被赋的值）的名称
        # 例如 s = socket.socket()，右边是一个 Call，提取出 "socket.socket"
        source_name = None
        if isinstance(node.value, ast.Call):
            source_name = self._get_call_name(node.value.func)
        elif isinstance(node.value, (ast.Name, ast.Attribute)):
            # 例如 s = socket
            source_name = self._get_call_name(node.value)

        # 如果右边确实是个 API 调用或模块名，把它和左边的变量名绑在一起记录下来
        if source_name:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assignments[target.id] = source_name
                    
        self.generic_visit(node)

    # --- 4. 解析调用并还原变量 ---
    def _get_call_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_call_name(node.value)
            if value:
                return f"{value}.{node.attr}"
        elif isinstance(node, ast.Call):
            # 【关键修复】处理链式调用：比如 request.urlopen().read()
            # 当发现前面的对象也是个调用时，递归进去提取源头的名字
            return self._get_call_name(node.func)
        return None

    def visit_Call(self, node):
        call_name = self._get_call_name(node.func)
        
        if call_name:
            # 【核心逻辑】：变量还原
            parts = call_name.split('.')
            base_var = parts[0] # 获取调用的基准对象，比如 s.connect() 的 "s"

            # 如果这个对象在我们的“记事本”里，说明它是个被赋值过的变量
            if base_var in self.assignments:
                resolved_base = self.assignments[base_var]
                # 把 s 替换成 socket.socket，拼装成 socket.socket.connect
                if len(parts) > 1:
                    call_name = f"{resolved_base}.{'.'.join(parts[1:])}"
                else:
                    call_name = resolved_base
            
            # 还原之后，重新获取 root_name 进行过滤判断
            root_name = call_name.split('.')[0]

            # 过滤逻辑
            if root_name in self.local_defs:
                pass  
            elif self.imports.get(root_name) == "__INTERNAL__":
                pass  
            elif root_name == self.package_name or self.imports.get(root_name, "").startswith(self.package_name):
                pass  
            elif root_name not in self.imports and root_name not in dir(__builtins__):
                pass
            else:
                resolved_root = self.imports.get(root_name, root_name)
                if resolved_root != root_name:
                    full_call = call_name.replace(root_name, resolved_root, 1)
                else:
                    full_call = call_name
                
                # 如果你需要保留内置函数(如 range, len)，把下面这两行删掉即可
                if root_name not in dir(__builtins__):
                    self.external_calls.add(full_call)

        self.generic_visit(node)

def extract_external_apis(package_path):
    package_path = Path(package_path)
    package_name = package_path.name
    all_external_calls = set()

    for root, dirs, files in os.walk(package_path):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read(), filename=filepath)
                    except SyntaxError:
                        continue
                
                analyzer = ExternalAPIAnalyzer(package_name)
                analyzer.visit(tree)
                all_external_calls.update(analyzer.external_calls)

    return all_external_calls

# 使用示例：
# 假设你的包目录路径是 /path/to/my_pkg
external_apis = extract_external_apis("/Data2/hxq/datasets/incremental_packages_subset/malicious/2024-02/update-request-0.0.1")
print("找到的外部 API 调用：")
for api in sorted(external_apis):
    print(api)