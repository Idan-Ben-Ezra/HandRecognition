import py_compile
def main():
    print("compiling")
    py_compile.compile("C:\\Pycharm Projects\\trying_model\\present.py", cfile="C:\\Pycharm Projects\\trying_model\\present.pyc")
    print("compiled")
if __name__ == "__main__":
        main()