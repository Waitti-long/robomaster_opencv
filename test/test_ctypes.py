import ctypes
so = ctypes.cdll.LoadLibrary
lib = so("../interfaces/lib_send.so")
lib.send_message(10, 20)