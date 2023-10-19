"""
NEF格式图片转换为png
"""
import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Progressbar
import os
import rawpy
import imageio
import threading

def convert_nef_to_png(input_path, output_path):
    try:
        with rawpy.imread(input_path) as raw:
            rgb = raw.postprocess()
            imageio.imwrite(output_path, rgb)
        # print("转换完成！")
    except IOError:
        print("无法打开或转换图像文件。")


def select_folder():
    folder_path = filedialog.askdirectory()
    entry_path.delete(0, tk.END)
    entry_path.insert(tk.END, folder_path)


def convert():
    input_folder = entry_path.get()
    if not os.path.exists(input_folder):
        print("选择的文件夹不存在！")
        return

    output_folder = input_folder + "_converted"
    os.makedirs(output_folder, exist_ok=True)

    files = [file_name for file_name in os.listdir(input_folder) if file_name.lower().endswith(".nef")]

    progress_bar['maximum'] = len(files)
    progress_bar['value'] = 0

    def convert_files():
        for i, file_name in enumerate(files):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name[:-4] + ".png")
            convert_nef_to_png(input_path, output_path)

            progress_bar['value'] = i + 1
            window.update()

        print("转换完成！输出文件夹：", output_folder)

    # 创建并启动后台线程
    thread = threading.Thread(target=convert_files)
    thread.start()


# 创建主窗口
window = tk.Tk()
window.title("NEF转换器")

# 选择文件夹按钮
btn_select_folder = tk.Button(window, text="选择文件夹", command=select_folder)
btn_select_folder.pack()

# 文件夹路径输入框
entry_path = tk.Entry(window, width=50)
entry_path.pack()

# 转换按钮
btn_convert = tk.Button(window, text="转换", command=convert)
btn_convert.pack()

# 进度条
progress_bar = Progressbar(window, orient=tk.HORIZONTAL, length=200, mode='determinate')
progress_bar.pack()

# 运行主循环
window.mainloop()
