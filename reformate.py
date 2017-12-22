origin_path = "/home/qiaoyang/codeData/molili/ProgramData/"
new_path = "/home/qiaoyang/bisheData/binary/"
head = "#include <iostream>\nusing namespace std\n\n"

def reformate(origin_path,new_path):
    dirs = os.listdir(origin_path)
    dirs.sort(key = lambda x:int(x))
    for dir in dirs:
        files = os.listdir(origin_path+dir)
        files.sort(key=lambda x: int(str(x).split('.')[0]))
        if (not os.path.isdir(new_path + dir)):
            os.mkdir(new_path + dir)
        count = 0
        for file in files:
            with open(origin_path+dir+'/'+file,'r') as f_read:
                code = f_read.read()
            with open(new_path+dir+'/'+str(count)+'.cpp','w') as f_write:
                f_write.write(head+code)
            count+=1