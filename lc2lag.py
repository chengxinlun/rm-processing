import os
from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model, Rmap_Model, Pmap_Model
import pickle


# Get the whole sid list
def get_total_sid_list():
    readfile = open("info_database/sid.pkl", "rb")
    sid_list = pickle.load(readfile)
    readfile.close()
    return sid_list


def obtainlag(file_con, file_line):
    # Load and fit continuum data
    c = get_data([file_con])
    cmod = Cont_Model(c)
    cmod.do_mcmc(threads=4)

    # Load and fit continuum and line data
    cy = get_data([file_con, file_line])
    cymod = Rmap_Model(cy)

    # Get time lag
    data_out = "cont-" + file_line.split(".")[0] + ".txt"
    cymod.do_mcmc(conthpd=cmod.hpd, threads=4, fchain=data_out)


def childprocess(pipe):
    for each_line in line_list:
        i=1
        while i<3:
            try:
                obtainlag(each_line+"-cont.txt", each_line+".txt")
                break
            except Exception as reason:
                i=i+1
        if i==3:
            print("Failed 3 times for "+str(each_sid)+" "+each_line+"\n")
            exception_list.append(str(reason))
    msg="0"
    os.write(pipe, msg.encode("UTF-8"))
    os.close(pipe)
    os._exit(0)


exception_list=list()
line_list=["Hbeta", "Mg2", "C4", "N5"]
sid_list = get_total_sid_list()
os.chdir("lc")
for each_sid in sid_list:
    os.chdir(str(each_sid))
    pipein, pipeout=os.pipe()
    newid=os.fork()
    if newid==0:
        os.close(pipein)
        childprocess(pipeout)
    else:
        os.close(pipeout)
        ret=os.read(pipein, 32)
        os.close(pipein)
    os.chdir("../")

os.chdir("../")
exception_handler=open("lag_error.log", "w")
for each in exception_list:
    exception_handler.write(str(each)+"\n")
exception_handler.close()
