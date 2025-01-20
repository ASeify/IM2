# version: 1.0.3
from tkinter import Tk, filedialog
from contextlib import contextmanager
import os
class Files_Handler:
    @staticmethod
    def select_files(file_type:'str', visible_types:'str', win_title:str='Select File', multiple_mod:'bool'=False):
        try:
            root = Tk()
            root.withdraw()
            root.call("wm", "attributes", ".", "-topmost", True)
            file_path = filedialog.askopenfilename(
                filetypes=[(file_type, visible_types)],
                title=win_title,
                multiple=multiple_mod)
            return file_path
        except Exception as e:
            print(e)
    
    @staticmethod
    def select_dir(win_title:str='Select Directory'):
        path = None
        try:
            root = Tk()
            root.withdraw()
            root.call("wm", "attributes", ".", "-topmost", True)
            path = filedialog.askdirectory(title=win_title)
            return path
        except Exception as e:
            return path

    @staticmethod
    def get_files_in_path(path:'str', types:str | list[str] = '*'):
        files = []
        files_list = next(os.walk(path), (None, None, []))[2]
        if types == '*':
            return files_list
        else:
            for item in files_list:
                if item.split(".")[-1] in types:
                    files.append(item)
        return files
    
    @staticmethod
    def make_dir(root_path: str, dir_name: str):
        new_path = root_path + dir_name
        if os.path.exists(new_path):
            return new_path + "/"
        try:
            os.mkdir(new_path)
            return new_path + "/"
        except:
            return False
    
    @staticmethod
    def create_new_file(path:str, name:str, ext:str, mode:str):
        if path[-1] != "/":
            path += "/"
            mode = mode
        return open(str(path + name + ext), mode)
    
    @staticmethod
    def get_file_info(inp_file):
        info = {}
        file_path = inp_file.name
        info['path'] = file_path[:file_path.rfind("/")] + "/"
        name_type = file_path.split(".")
        type = "." + name_type[-1]
        if len(name_type) > 2:
            name_type = file_path.split("/")[-1]
            info['name'] = name_type[:name_type.rfind(".")]
        else:
            info['name'] = name_type[0].split("/")[-1]
        info['type'] = type
        info['mode'] = inp_file.mode
        
        return info
   
    @staticmethod
    def get_file_path_info(file_path:str):
        info = {}
        info['path'] = file_path[:file_path.rfind("/")] + "/"
        name_type = file_path.split(".")
        type = "." + name_type[-1]
        if len(name_type) > 2:
            name_type = file_path.split("/")[-1]
            info['name'] = name_type[:name_type.rfind(".")]
        else:
            info['name'] = name_type[0].split("/")[-1]
        info['type'] = type        
        return info
    
    @staticmethod
    def get_dirs_in_path(file_path:str):
        dir_list = [x[0] for x in os.walk(file_path)]
        return dir_list
    
    @staticmethod
    @contextmanager
    def tk_windows_timer(timeout=600):
        root_timer = Tk() # default root
        root_timer.withdraw() # remove from the screen

        # destroy all widgets in `timeout` seconds
        func_id = root_timer.after(int(1000*timeout), root_timer.quit)
        try:
            yield root_timer
        finally: # cleanup
            root_timer.after_cancel(func_id) # cancel callback
            root_timer.destroy()
    pass