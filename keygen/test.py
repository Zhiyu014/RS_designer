from hashlib import md5
mac = input('请输入机器码：')
code = md5(mac.encode(encoding='UTF-8')).hexdigest()
print(code)
input('按任意键关闭进程：')

# Compile: pyinstaller -F -p C:\Users\zhiyu\.vscode\extensions\ms-python.vscode-pylance-2023.1.20\dist\typeshed-fallback\stdlib\hashlib.pyi test.py