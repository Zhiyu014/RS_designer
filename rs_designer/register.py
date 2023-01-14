from wmi import WMI
from hashlib import md5
from os.path import join,dirname,exists

s = WMI()

class Register:
    def __init__(self):
        pass

    #cpu序列号
    def get_CPU_info(self):
        cpu = []
        cp = s.Win32_Processor()
        for u in cp:
            cpu.append(
                {
                    "Name": u.Name,
                    "Serial Number": u.ProcessorId,
                    "CoreNum": u.NumberOfCores
                }
            )
        return cpu
 
    #硬盘序列号
    def get_disk_info(self):
        disk = []
        for pd in s.Win32_DiskDrive():
            disk.append(
                {
                    "Serial": pd.SerialNumber.lstrip().rstrip(),
                    "ID": pd.deviceid,
                    "Caption": pd.Caption,
                    "size": str(int(float(pd.Size)/1024/1024/1024))
                }
            )
        return disk
 
    #mac地址（包括虚拟机的）
    def get_network_info(self):
        network = []
        for nw in s.Win32_NetworkAdapterConfiguration():
            if nw.MacAddress != None:
                network.append(
                    {
                        "MAC": nw.MacAddress,
                        "ip": nw.IPAddress
                    }
                )
        return network
 
    #主板序列号
    def get_mainboard_info(self):
        mainboard = []
        for board_id in s.Win32_BaseBoard():
            mainboard.append(board_id.SerialNumber.strip().strip('.'))
        return mainboard

    def get_mach_num(self):
        a = self.get_network_info()
        b = self.get_CPU_info()
        c = self.get_disk_info()
        d = self.get_mainboard_info()
        macc_str = ""
        macc_str = macc_str + a[0]['MAC'] + b[0]['Serial Number'] + c[0]['Serial'] + d[0]
        macc_str = macc_str.replace(':','')
        # macc_code = md5(macc_str.encode(encoding='UTF-8')).hexdigest()
        return macc_str

    def regist(self,key):
        macc_str = self.get_mach_num()
        code = md5(macc_str.encode(encoding='UTF-8')).hexdigest().encode(encoding='UTF-8')
        if code != 0 and key != 0:
            key_decrypted = key.encode(encoding='UTF-8')
            if code == key_decrypted:
                with open(join(dirname(__file__),'register.lic'), 'w') as f:
                    f.write(key)
                    f.close()
                return True
            else:
                return False
        else:
            return False


    def checkAuthored(self):
        macc_str = self.get_mach_num() 
        code = md5(macc_str.encode(encoding='UTF-8')).hexdigest().encode(encoding='UTF-8')
        if not exists(join(dirname(__file__),'register.lic')):
            return False
        else:
            try:
                f = open(join(dirname(__file__),'register.lic'),'r')
                key = f.read()
                if key and key.encode(encoding='UTF-8') == code:
                    return True
                else:
                    return False
            except:
                return False


# if __name__ == '__main__':
#     reg = Register()
#     mac = reg.get_mach_num()
#     code = md5(mac.encode(encoding='UTF-8')).hexdigest()
#     with open(join(dirname(__file__),'register.lic'), 'w') as f:
#         f.write(code)
#         f.close()
#     p = input()