import os
from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model, Rmap_Model, Pmap_Model


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

    cy.plot()

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
            print("Failed 3 times for "+each_target+" "+each_line+"\n")
            exception_list.append(str(reason))
    msg="0"
    os.write(pipe, msg.encode("UTF-8"))
    os._exit(0)


exception_list=list()
line_list=["Hbeta", "Mg2", "C4", "N5"]
os.chdir("lc")
target_list=os.listdir(os.getcwd())
for each_target in target_list:
    os.chdir(each_target)
    pipein, pipeout=os.pipe()
    newid=os.fork()
    if newid==0:
        childprocess(pipeout)
    else:
        ret=os.read(pipein, 32)
    os.chdir("../")

os.chdir("../")
exception_handler=open("lag_error.log", "w")
for each in exception_list:
    exception_handler.write(str(each)+"\n")
exception_handler.close()
